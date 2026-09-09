# NPO representation-guided effective batches

This directory implements the experiment in which cached representations from
the initial LLaMA-3.1-8B TOFU Full model determine the composition of NPO
effective batches. It does not reorder examples inside one microbatch and does
not refresh representations during unlearning.

The four first-stage arms are:

- `R`: random grouping without replacement.
- `S`: greedily place similar examples in one effective batch.
- `D`: greedily place dissimilar examples in one effective batch.
- `P`: permute the sample-to-feature mapping, then run the `S` algorithm.

Every arm covers every forget row exactly once per epoch. The global effective
batch is 20: two ranks, per-device microbatch 1, and ten accumulation steps.
Every epoch must divide into complete accumulation windows; samples are never
dropped, duplicated, or carried across an epoch boundary. Retain row ids are
generated once per seed and are identical at corresponding optimizer steps
across arms.

## 1. Server environment

Use the repository's pinned environment (`transformers==4.45.1`,
`accelerate==0.34.2`, `deepspeed==0.15.4`, `torch==2.4.1`). Accept the LLaMA
license and authenticate on the server with `hf auth login` or a secret
`HF_TOKEN`. Do not put a token in a script or config file.

For an offline server, copy the complete TOFU repository directory and the Full
model snapshot to local storage. Passing the absolute TOFU directory through
`--dataset` selects `<config>.json` directly, so the copied Dataset Card does
not need to expose Hub `BuilderConfig` metadata. Pass the local model directory
through `--model`, and omit revision arguments for both local paths.

All commands below run from the repository root. The local laptop is suitable
for manifest tests but is not expected to load the model.

## 2. Extract features once

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/representation_batching/extract_tofu_features.py \
  --model open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
  --dataset locuslab/TOFU \
  --dataset-config forget05 \
  --batch-size 2 \
  --block two_thirds \
  --output-dir artifacts/representation_batching/features-forget05
```

The extractor uses the same LLaMA chat template assumptions as the training
configuration. It runs the complete question-answer sequence once: causal
masking lets it read question positions without seeing future answer tokens,
and the same forward pass supplies initial answer NLL. It stores only six pooled
vectors per row, not all token activations. `--max-length` is a guard: the
extractor fails instead of truncating tokens that the repository training path
would keep. Pass immutable `--model-revision` and `--dataset-revision` values
for a final run; the resolved model and tokenizer commit ids are recorded.

For LLaMA-3.1-8B (32 decoder blocks), `two_thirds` resolves to one-indexed block
22. The primary first-stage key is:

```text
block_22_question_last
```

Before continuing, inspect `feature_manifest.json`. The script fails if the
rendered-template tokenization disagrees with `apply_chat_template`, if the
question or answer span is absent, or if a sequence exceeds the configured
length guard.

## 3. Build manifests

First confirm the actual retain95 row count on the server. For standard TOFU it
is expected to be 3800, but pass the observed value rather than trusting this
note. Build one manifest directory per paired seed:

```bash
for seed in 0 1 2; do
  python scripts/representation_batching/build_batch_manifests.py \
    --features artifacts/representation_batching/features-forget05/features.npz \
    --feature-key block_22_question_last \
    --output-dir artifacts/representation_batching/seed-${seed} \
    --retain-size 3800 \
    --effective-batch-size 20 \
    --world-size 2 \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 10 \
    --num-epochs 3 \
    --seed "${seed}"
done
```

Each directory contains `R.jsonl`, `S.jsonl`, `D.jsonl`, `P.jsonl`,
`batch_stats.csv`, and `run_manifest.json`. Compare the within-batch cosine
distribution before launching training. The CSV also carries mean question and
answer token lengths, initial answer NLL, and author composition for each
effective batch. If S, R, and D do not differ, stop and check the selected
representation rather than running the model grid.

The second-stage batch-order experiment is already supported. Keep a partition
method fixed and rebuild with `--batch-order similar` or
`--batch-order diverse`. Those files are named, for example,
`R_order-similar.jsonl`. Do this only after the first-stage grouping comparison.

## 4. Smoke and train

The launcher uses exactly two Accelerate processes with DeepSpeed ZeRO-3. It
does not use the older NPO shell script, whose direct `python src/train.py` call
does not launch the configured processes.

Run a no-save smoke with a complete manifest and stop after two optimizer
steps. The trainer then audits the consumed prefix while retaining the full
manifest contract:

```bash
scripts/representation_batching/run_npo_representation.sh \
  --manifest artifacts/representation_batching/seed-0/R.jsonl \
  --gpu 0,1 \
  --seed 0 \
  --max-steps 2 \
  --no-save
```

Then run the four paired arms for each seed:

```bash
for seed in 0 1 2; do
  for arm in R S D P; do
    scripts/representation_batching/run_npo_representation.sh \
      --manifest "artifacts/representation_batching/seed-${seed}/${arm}.jsonl" \
      --gpu 0,1 \
      --model-revision MODEL_COMMIT \
      --dataset-revision DATASET_COMMIT \
      --seed "${seed}"
  done
done
```

The default is full-parameter NPO plus retain NLL, beta 0.1, learning rate
2e-5, three epochs, constant schedule, bf16, and gradient checkpointing. The
launcher saves only the final model. Set `WANDB_MODE=online` if desired; offline
logging is the default.

Run the repository's TOFU evaluator separately on one GPU after training. The
current custom evaluator intentionally skips multi-process execution. Full,
Retain95, R/S/D/P, and any matched-forgetting checkpoints must use the same
evaluator code, prompts, and decoding configuration.

## 5. Required audit checks

Training aborts when the manifest disagrees with runtime world size, per-device
batch, accumulation, epoch count, forget row ids, or retain row count. The
ordered question-answer content is hashed as well, so a changed TOFU revision or
row order cannot silently reuse stale representations. The
custom dataloader is not passed through `accelerator.prepare`, because it has
already been split by global rank; Trainer still moves tensors to the correct
device. Each process compares the row ids emitted by the collator with its
planned microbatch and writes separate planned and observed JSONL logs. Rank 0
also archives the resolved Hydra config, package versions, git state, tracked
diff, and a copy of the experiment-specific source files.

At smoke time verify in both rank logs:

- ten local microsteps per optimizer step;
- no repeated or missing forget ids;
- planned and observed rank schedules agree exactly;
- equal local dataloader lengths;
- NPO forget loss near `2 log(2) / 0.1 = 13.8629` at initialization, before
  adding retain loss;
- NPO weight near 1 at initialization;
- the feature file hash and resolved Hydra config are archived with the run.

If the 8B active model plus frozen reference does not fit under this ZeRO-3
configuration, record the peak-memory failure and revise the resource setting.
Do not silently switch one arm to LoRA, CPU offload, another optimizer, or a
different effective batch.

## 6. Unified results: one HTML entry point

From the repository root, refresh the report after training or evaluation:

```bash
python3 scripts/representation_batching/summarize_results.py \
  --config configs/analysis/representation_batching.json
```

Open `reports/representation_batching/index.html`. This static page contains the
planned seed/arm coverage, a searchable run table, final TOFU metrics, batch
similarity diagnostics, per-run training curves, within-protocol mean/sample
standard deviation, and same-seed differences against `R/random`. Raw exports
are `runs.csv`, `coverage.csv`, `aggregates.csv`, `paired_deltas.csv`, and
`results.json`. Re-run the command to refresh; the page is not a live monitor.
No GPU, model weights, W&B login, or web service is required for reporting.
New runs can be read with the Python standard library. Legacy YAML-only runs
need PyYAML in the reporting environment.

To scan scattered training directories, repeat `--root` (these override roots in
config). Source files are not moved or modified:

```bash
python3 scripts/representation_batching/summarize_results.py \
  --config configs/analysis/representation_batching.json \
  --root /absolute/server/experiment-A \
  --root /absolute/server/experiment-B \
  --out reports/representation_batching
```

Paths in the JSON config resolve relative to the config file; command-line
paths resolve relative to the current directory. For evaluation results stored
elsewhere, add explicit associations to `evaluations` in the config:

```json
"evaluations": [
  {
    "run_dir": "/absolute/path/to/run",
    "summary": "/absolute/path/to/separate/eval/TOFU_SUMMARY.json"
  }
]
```

The default automatic evaluation location is `<run>/evals/TOFU_SUMMARY.json`.
Multiple final summaries for one run are flagged as ambiguous; the report does
not choose the largest metric or most recent file. It does not infer identity
from timestamp/run-name similarity or mix intermediate checkpoint metrics into
final-model comparisons.

For future completed runs, put evaluation outputs there directly:

```bash
python3 scripts/representation_batching/evaluate_run.py \
  --run /absolute/path/to/run \
  --retain-logs /absolute/path/to/matched/retain95/TOFU_EVAL.json \
  --gpu 0
```

This launches the existing single-GPU TOFU evaluator with the run's model type,
Forget/Holdout splits, and a saved final model. It recomputes metrics
(`overwrite=true`) so cached evaluations from other settings cannot be silently
relabelled. `--dry-run` prints the command without loading the model. The current
helper uses the repository's default TOFU evaluation configuration. It accepts
`--dataset`, `--classifier-model`, and `--batch-size` for local evaluation paths
and batch size. Custom prompts or other metric overrides require calling
`src/eval.py` directly with those overrides and `paths.output_dir=<run>/evals`.
The supplied Retain logs must use the same model/split/evaluation protocol.

### Read the report in this order

1. **Coverage/status:** `limited` is a max-steps run, excluded from final
   comparisons. `incomplete_or_unknown` means there is insufficient completion
   evidence, not proof that a process is currently running or failed.
2. **Audit:** planned and observed microbatches must agree on every rank, and
   archived plans must reproduce the actual batch manifest. An exhausted log
   alone is not proof the last optimizer update finished.
3. **Grouping and order:** inspect within-batch cosine for grouping experiments;
   newly generated manifests additionally store `previous_batch_cosine` for
   adjacent-batch order. This uses true representations and excludes the first
   batch of each epoch. Older manifests show this field as missing. Compare
   question/answer lengths and initial sequence NLL as potential covariates.
4. **Trajectories:** inspect forget/retain losses and NPO weight/near-zero
   fraction. These are training diagnostics, not deletion scores. Diagnostics
   are measured before the incoming optimizer update, including the last point.
5. **Final outcomes:** compare matching protocols and paired seeds, considering
   Forget quality, utility, fluency, and matched Retain exact memorization
   together. FQ is a KS p-value, not a forgetting percentage; its difference is
   not a calibrated effect size. No composite ranking is generated.

For a batch-order comparison, set `expected_arms` to e.g.
`["S/random", "S/similar", "S/diverse"]` and `reference_arm` to `"S/random"`.
Keep `expected_seeds` consistent with the actual experiment plan. Coverage is a
coarse inventory across all scanned settings, while statistics are separated by
training and evaluation protocol fingerprints. Duplicate seed/arm runs within a
protocol are displayed but excluded from aggregation and paired differences
until you explicitly select the intended run directories.

### New output files and legacy compatibility

New training runs persist `resolved_config.json`, `training_diagnostics.jsonl`,
and `training_status.json` independently of model saving, including `--no-save`.
A `completed` training marker means the trainer returned after full schedule
consumption, not that model saving or final evaluation succeeded. Process kills
may leave `started`; the report treats that as unknown rather than live status.
A positive `max_steps` is conservatively always classified as `limited`.

Standalone evaluation now writes `evaluation_provenance.json` with resolved
settings, evaluator/data/model source hashes, Retain log hash, and completion
state. Only a completed evaluation attached to the run's final model with all
configured metrics present can enter formal aggregation. Reusing old cached
metrics without matching provenance leaves it unverified. Old summary JSONs
still appear in the ledger but do not automatically enter paired comparisons.
Full/Retain baseline files in the config are shown as references only; protocol
compatibility is not assumed. Missing metrics are never replaced with zero.

To inspect server results on a laptop, either copy the generated report folder
(HTML/CSV/JSON work offline), or copy each run's small config/status/diagnostic
files, `batch_audit`, `trainer_state.json` if available, and `evals` for rebuilding
locally. Weights are unnecessary. Archived original output paths allow final
model provenance to survive a move; links to uncopied source files naturally
remain unavailable. Different absolute dataset/model paths conservatively form
separate training cohorts. No remote connection or sync is performed by the
reporting script.

## 7. 输入一个 seed，自动训练、评估、画图

在项目训练环境中，从仓库根目录运行：

```bash
bash scripts/representation_batching/run_seed.sh 1
```

脚本依次完成：复用初始特征 → 生成 seed 1 的 R/S/D/P 清单 → R 训练与评估 →
S 训练与评估 → D 训练与评估 → P 训练与评估 → 更新总览、跨 seed 表格和四指标图。
每一组均从同一份初始 Full 模型开始。训练使用两张 GPU，评估使用单张 GPU。
只想预览路径与命令时，在末尾加 `--dry-run`；它不会创建运行目录或启动模型。

### 首次运行的路径配置

集中配置文件是 `configs/analysis/representation_pipeline.json`。
**这个文件里的相对路径统一相对于仓库根目录**，与上节报告配置的路径规则不同。
通常只需要首次确认路径，后续只输入 seed：

- `features: null`：先查默认 `artifacts/representation_batching/features-forget05/features.npz`，
  找不到则在 artifacts 中找唯一一份特征文件。多个候选时请填写原文件路径。
- `model: null` / `dataset: null`：沿用特征文件旁 `feature_manifest.json` 记录的模型和数据来源。
  本地路径必须存在；搬过目录时，在这里填写新路径。不会自动切换模型或重新提取特征。
- `retain_logs`：配套 Retain95 的 `TOFU_EVAL.json`，不要填写 SUMMARY。
- `classifier_model`：本地 gibberish-detector 目录，默认 `models/gibberish-detector`。
- `retain_size: 3800`：沿用当前 forget05/retain95 设置，训练器仍会核对实际数据数量。
- `training_gpus: "0,1"` / `evaluation_gpu: "0"`：按服务器空闲 GPU 修改。
- `evaluation_batch_size: 32`、`learning_rate: 0.00002`：四组共用；改变后属于新的实验配置。

可另存一份自己的配置，并使用 `--config /path/to/pipeline.json`。
模型和 TOFU 可以使用特征元数据中的 Hub id；本地离线评估会把所有 TOFU 数据配置
统一指向 `dataset`。分类器和 Retain 参考日志需要在本地准备好。
本流程固定使用当前的 LLaMA-3.1-8B / forget05、effective batch 20、3 epochs、
random batch order；它是四种**分组方式**的配对 seed 实验。

### 断点继续和输出位置

同一个 seed 中途失败后，重新运行原命令即可。已完成且保存模型的训练不会重复；
脚本同时检查全部模型分片和命令成功后写出的 `model_save_complete.json`，避免跳过未写完的模型。
已完成且来源与四项指标核验通过的评估也会跳过。评估失败会重做该组评估。
未完成的训练从 Full 模型开始，在新的 `attempt-*` 目录中重跑，旧记录保留。
**这里是阶段级继续，不是恢复中断训练的 optimizer 状态。**

默认目录：

```text
artifacts/representation_batching/seed_runs/seed-1/   # 四份清单
saves/unlearn/tofu/forget05/Llama-3.1-8B-Instruct/representation_npo/seed_runs/seed-1/
  pipeline_state.json                              # 四组进度、设置与运行路径
  logs/                                            # 每阶段终端日志
  R/attempt-0001/                                   # 模型、训练诊断、evals/
  S/attempt-0001/
  D/attempt-0001/
  P/attempt-0001/
reports/representation_batching/
  index.html                                       # 原审计总览
  seed_comparison.html                             # 一张表 + 四指标图
  seed_comparison.csv                              # seed × metric，列为 R/S/D/P
  seed_runs.csv                                    # 每次运行的来源与协议
  seed_comparison.png / .pdf / .svg
```

同一 seed 有进程锁，避免重复启动。重跑期间若配置、特征、Retain 日志或相关代码改变，
脚本会停止，而不会混接两个协议；要做新条件，请另设 `output_root` 和 `manifest_root`。
旧版时间戳目录仍会进入汇总，但不自动被接管或视为本脚本的完成阶段。
要依次运行 seed 1、2：

```bash
for seed in 1 2; do
  bash scripts/representation_batching/run_seed.sh "$seed" || break
done
```

### 单独生成跨 seed 表格和图

```bash
python3 scripts/representation_batching/compare_seeds.py
```

默认扫描 `saves/unlearn`。也可以重复 `--root` 指定多个目录，或用
`--results reports/representation_batching/results.json` 读取已生成的汇总。
`--seeds 0 1 2` 会把尚无记录的 seed 也显示出来。

每个指标一幅子图，横轴为 seed，颜色对应 R/S/D/P。只有同一已核验协议内的点才连线；
旧版/未核验结果用空心点展示。seed 编号只是独立重复的标签，不代表优化进度。
缺失值留空、不补零；同一个 seed/arm 的多个完整运行会标记为重复，不自动选最新或最佳。
为明确选择重复运行，可以将每个选中的运行目录分别作为 `--root` 传入。
图使用 Matplotlib（项目 requirements 已包含），没有图库时仍会导出表格并提示缺失依赖。

只拿到用户手填的数字时，可使用 `--metrics-csv /path/to/metrics.csv`。
CSV 列为 `seed,arm,exact_memorization,forget_Q_A_gibberish,forget_quality,model_utility`。
这种来源始终标记为 `user_reported`，不冒充已核验的服务器结果；与同一 seed/arm 的
自动采集结果重复时，该单元格留空，需先选择来源。
