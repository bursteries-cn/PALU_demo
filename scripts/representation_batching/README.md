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
2e-5, three epochs, constant schedule, bf16, gradient checkpointing, and global
gradient clipping at 1.0. The DeepSpeed config must keep
`gradient_clipping: "auto"`; Accelerate delegates clipping to DeepSpeed, whose
default for an omitted key is zero. The launcher saves only the final model.
Set `WANDB_MODE=online` if desired; offline logging is the default.

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
- `runtime_contract.json` records matching `max_grad_norm` and resolved
  `deepspeed_gradient_clipping` values.

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
终端显示 `[1/4] R` 到 `[4/4] P`，以及当前是在训练还是评估。
单组训练/保存检查/评估失败会记录原因并继续其余组；四组全部尝试后，若仍有失败，
脚本返回非零退出码并列出失败项。清单生成失败或手动中断则立即停止。
只想预览路径与命令时，在末尾加 `--dry-run`；它不会创建运行目录或启动模型。

### 修改训练轮数

默认仍为 3 个 epoch。通过 `--epochs` 同时改变四组清单的轮数和训练轮数：

```bash
# 先检查路径和命令；不加载模型、不创建运行目录
bash scripts/representation_batching/run_seed.sh 0 --epochs 10 --gpu 0,1 --dry-run

# seed 0：R/S/D/P 各自从原始 Full 模型训练 10 个 epoch，再分别评估
bash scripts/representation_batching/run_seed.sh 0 --epochs 10 --gpu 0,1
```

`--epochs` 优先于 `configs/analysis/representation_pipeline.json` 中的
`num_epochs`，要求正整数。不提供时沿用配置（默认 3）。对于非 3 轮实验，
流水线会在配置的 `output_root`、`manifest_root` 和 `report_dir` 后分别增加
`epochs-N` 子目录，例如 `seed_runs/epochs-10/seed-0/`。自动生成的报告只扫描
该轮数的输出目录，避免同一个 seed 的 3 轮和 10 轮结果被当作重复运行。
需要全局台账时，可另外用 `summarize_results.py --root saves/unlearn` 扫描所有结果。

这会从 Full 重新开始完整的 10 轮实验，不是在旧 3 轮模型上继续 7 轮。
特征文件继续复用。200 条 Forget、有效 batch 20 时，每轮 10 次参数更新，
3/10/20 轮分别是 30/100/200 次更新；增大轮数不会增加独立样本数量。
当前仍只评估最终模型，不自动提供每个 epoch 的评估曲线。

若单独调用训练启动器，也可以使用 `--epochs 10`，但必须先用
`build_batch_manifests.py --num-epochs 10` 生成新的完整清单，并通过
`--manifest` 指向该清单；训练器会拒绝轮数不匹配。单独启动器沿用自己的
`--output-root` / `--output-dir` 规则，自动按轮数隔离目录是 `run_seed.sh` 的功能。

### 在不同 GPU 组上手动并行多个 seed

在两个终端（或两个 tmux 窗口）分别启动：

```bash
# 终端 1：seed 1，训练 GPU 0、1，评估 GPU 0
bash scripts/representation_batching/run_seed.sh 1 --gpu 0,1

# 终端 2：seed 2，训练 GPU 2、3，评估 GPU 2
bash scripts/representation_batching/run_seed.sh 2 --gpu 2,3
```

参数含义：

- `--gpu`（别名 `--gpus` / `--training-gpus`）：两张 GPU 的编号，覆盖配置文件。
- `--eval-gpu`：单张评估 GPU；指定 `--gpu` 且不提供此参数时，自动使用该组的第一张，
  例如 `--gpu 2,3` 默认评估 GPU 2，不会继续使用配置文件里的 GPU 0。
- `--port`（别名 `--main-process-port`）：分布式通信端口，默认 `29500 + 两张训练 GPU 中较小的编号`。
  上述两组分别使用 29500、29502；若端口已被其他作业占用，可显式换一个。
  此参数传给 Accelerate 的 [`--main_process_port`](https://huggingface.co/docs/accelerate/v0.34.2/en/package_reference/cli#accelerate-launch)。
- 默认在单卡评估成功、四项指标和来源核验通过后删除最终模型权重，保留训练审计、日志、
  `TOFU_EVAL.json`、`TOFU_SUMMARY.json` 和汇总结果。需要保留某个 seed 的四组模型时加
  `--keep-model`；`--no-save` / `--no-keep-model` 可显式要求评估后删除，也可在集中配置中
  设置 `"keep_model_weights": true`。

```bash
bash scripts/representation_batching/run_seed.sh 3 --gpu 4,5 --eval-gpu 5 --port 29605
```

这些编号直接用于 `CUDA_VISIBLE_DEVICES`。手动并行时为各任务分配不重叠的 GPU；
每个 seed 内的 R/S/D/P 仍串行执行，训练仍固定使用两卡与 effective batch 20。
同一 seed 的重复启动会被进程锁阻止，不支持用多个 `run_seed.sh` 实例拆分同一 seed。
GPU 编号和端口变化不会阻止阶段续跑；实际值会记录在 `pipeline_state.json` 的 `runtime` 中。
多个 seed 完成时会依次获取总报告写入锁，避免 CSV/图被同时覆盖；该锁不会串行化训练。

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
- `keep_model_weights: false`：评估成功后删除最终权重。该项只控制产物保留，不改变训练或评估数值协议。

可另存一份自己的配置，并使用 `--config /path/to/pipeline.json`。
模型和 TOFU 可以使用特征元数据中的 Hub id；本地离线评估会把所有 TOFU 数据配置
统一指向 `dataset`。分类器和 Retain 参考日志需要在本地准备好。
本流程固定使用当前的 LLaMA-3.1-8B / forget05、effective batch 20、3 epochs、
random batch order；它是四种**分组方式**的配对 seed 实验。

### 断点继续和输出位置

同一个 seed 中途失败后，重新运行原命令即可。已完成且有完整模型、或已有评估与权重清理标记的训练不会重复；
脚本同时检查全部模型分片和命令成功后写出的 `model_save_complete.json`，避免跳过未写完的模型。
已完成且来源与四项指标核验通过的评估也会跳过。评估失败会重做该组评估。
默认情况下，权重只保留到评估核验成功；随后写入 `model_weights_discarded.json` 并删除权重分片。
该标记与完整评估共同用于断点继续，因此重跑命令不会重新训练已经成功且已清理权重的组。
评估失败时不会删除权重，以便下次只重做评估。旧运行若没有记录这一保留策略，不会被自动清理。
评估 JSON 缺失或损坏不会把已完成训练误判为需要重训。
旧版若出现 `评估返回，但四个完整指标/来源核验未通过`，可能是来源记录丢失模型路径：
`get_model()` 会原地移除 `pretrained_model_name_or_path` 和 `torch_dtype`，
旧版却在调用它之后才记录配置。新版在加载模型前保存完整请求配置。
对旧结果，脚本仅在旧加载器哈希已知、同次评估的 `.hydra/config.yaml` 与
tokenizer 路径及其余模型设置一致、归档不晚于完成记录时恢复缺失字段。
四个指标完整且通过这些检查即可跳过旧评估，`evaluation_note` 会说明恢复来源；
原始来源记录不被改写。证据不足时只补做评估。请保留各组 `evals/.hydra/`。
未完成的训练从 Full 模型开始，在新的 `attempt-*` 目录中重跑，旧记录保留。
**这里是阶段级继续，不是恢复中断训练的 optimizer 状态。**

每个 seed 的入口是 **`seed_results.csv` / `seed_results.json`**：始终有 R/S/D/P 四行，
包括四个指标、各组状态、错误原因、模型目录，以及原始 SUMMARY/EVAL 的路径。
每个阶段切换和每组结束都会更新，即使 R 失败也会保留 S/D/P 的结果。
失败、未完成的组不填写指标，不拿残留的部分评估冒充完整结果。
原始评估继续保存在各组 `evals/` 下，模型不会为生成结果表而额外复制。

默认目录：

```text
artifacts/representation_batching/seed_runs/seed-1/   # 四份清单
saves/unlearn/tofu/forget05/Llama-3.1-8B-Instruct/representation_npo/seed_runs/seed-1/
  pipeline_state.json                              # 四组进度、设置与运行路径
  seed_results.csv / seed_results.json              # 本 seed 四组结果与文件入口，实时更新
  logs/                                            # 每阶段终端日志
  R/attempt-0001/                                   # 训练诊断、evals/；默认评估后移除权重
    model_weights_discarded.json                     # 权重清理完成记录
    evals/TOFU_SUMMARY.json                         # EM / Fluency / FQ / MU 四项汇总
    evals/TOFU_EVAL.json                            # 指标计算缓存与样本级评估记录
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

同一 seed 有进程锁，避免重复启动。重跑期间若配置、特征、Retain 日志或训练/评估计算代码改变，
脚本会停止，而不会混接两个协议；要做新条件，请另设 `output_root` 和 `manifest_root`。
仅流程控制或报告代码修复不再阻止续跑。旧版 `a5d9198` / `b1d2b1e` 的进度可自动兼容：
脚本会用 Git 中该版本重建旧签名，并确认实验设置和计算代码没有改变，才保留原进度。
如果服务器缺少这个提交的 Git 历史，先获取该分支完整历史；不要手动删除状态文件强行跳过检查。
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

新旧目录**不需要改名或搬迁**。扫描器递归查找 `batch_audit/`，从归档清单读取
seed 和 R/S/D/P，而不是解析外层文件夹名称，因此以下两种布局会同时识别：

```text
representation_npo/representation_npo_R_seed0_时间戳/batch_audit/
representation_npo/seed_runs/seed-2/R/attempt-0001/batch_audit/
```

各运行的 `evals/TOFU_SUMMARY.json`、来源记录及训练归档应一同保留。
默认扫描 `saves/unlearn` 已覆盖这两种默认布局。模型权重不参与报表扫描。
如果曾把结果放到默认范围之外，可在同一次汇总中重复指定 `--root`：

```bash
python3 scripts/representation_batching/compare_seeds.py --root /path/to/old_runs --root /path/to/new_runs
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
