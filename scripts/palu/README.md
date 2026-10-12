# PALU 后续实验基线

分支 `experiment/palu-diagnostics-baseline` 从原对比分支的 `cca8424` 建立。训练固定采用官方全词表均值的完整梯度，每个 seed 只训练一次。原实验代码及双组汇总在 `experiment/palu-mean-gradient-ablation` 分支保留。

## 一键启动与参数

在 `scripts/palu/run.sh` 顶部统一设置参数，然后运行：

```bash
bash scripts/palu/run.sh
```

主要设置：`EXPERIMENT`（实验内容名称）、`OUTPUT_ROOT`（baseline 保存根目录）、`GPU_IDS`、`SEEDS`、`LR`、`ALPHA`、`GAMMA`、`TOP_K`、`FIRST_N`、`EPOCHS`、`MAX_STEPS`、`BATCH_SIZE`、`GRAD_ACCUM`。也可以临时覆盖：

```bash
LR=1e-5 ALPHA=0.3 bash scripts/palu/run.sh
EXPERIMENT=batch8_first_sensitive SEEDS="0 1 2" bash scripts/palu/run.sh
bash scripts/palu/run.sh --dry-run
SEEDS="0" MAX_STEPS=2 PROBE_EVERY=1 bash scripts/palu/run.sh --skip-eval
```

默认 seed 为 `0 1 2`，GPU 为 `0,1,2,3`，学习率 `2e-5`，alpha `0.2`，gamma `1.0`，top-K `5000`，first-N `3`，训练 `10` epochs。`GRAD_ACCUM` 留空时自动选择全局有效 batch 32；每卡 batch × 卡数必须整除 32，否则显式设置累计步数。这些是本地实验起点，不是对所有任务验证过的最优参数。

需要核对的路径：

- `MODEL_PATH`：初始 Full 模型目录或 Hub ID。
- `RETAIN_MODEL_PATH`：Retain95 模型完整权重，供表征诊断使用。
- `RETAIN_LOGS`：匹配的 Retain 模型 `TOFU_EVAL.json`，供 FQ 评估使用。
- `DATASET_PATH`：服务器默认 `/mnt/sda/cr/LLM_unlearning/datset/TOFU`，保留实际目录拼写。
- `CLASSIFIER_MODEL`：流畅度分类器目录。

Retain95 权重默认读取仓库下已下载的本地模型目录：

```bash
RETAIN_MODEL_PATH="${RETAIN_MODEL_PATH:-${PALU_ROOT}/models/tofu_${MODEL}_${RETAIN_SPLIT}}"
```

服务器默认目录为 `/mnt/sda/cr/LLM_unlearning/PALU/models/tofu_Llama-3.1-8B-Instruct_retain95`，可通过 `RETAIN_MODEL_PATH` 覆盖。该目录必须包含完整权重、索引、配置和 tokenizer。评估 JSON 不能替代权重；仅设置离线模式也不能补齐缺失文件。参考模型、评估 JSON 和 Forget/Retain split 应匹配。

## 四条训练损失曲线

`plots/loss_curves.png` 和 PDF 含四条折线：Total、Forget、Retain NLL、Common-token KL。三个分量均乘以实际训练权重，满足：

```text
total = gamma * sensitive_forget_loss
      + alpha * retain_NLL
      + alpha * common_token_KL
```

每个优化步记录一次，聚合该步实际的梯度累计 microbatch 和全部训练进程。各图点取同一步可用 seed 的均值，T 型误差棒为点态 95% Student-t 置信区间；仅一个 seed 时不伪造置信区间。图标题和 `curve_statistics.csv` 记录实际 n 与计划 n。每个 seed 完成后自动重画，训练未结束时可能缺少部分 seed，最终以全体 seed 完成的图为准。

不平滑、不插值。置信区间描述训练种子变化，不包含固定探针样本选择的不确定性，也不是整条曲线的同时置信带。原始未加权分量也保存在历史文件中。

## Retain95 表征诊断

默认从实际训练数据固定抽取 32 个 Forget 和 32 个 Retain 样本，同一集合跨 seed 共用。step 0、每 5 步以及最后一步测量一次；`PROBE_EVERY`、`PROBE_FORGET_SAMPLES`、`PROBE_RETAIN_SAMPLES`、`PROBE_SEED`、`PROBE_TEMPERATURE` 可修改，样本数支持 `all`。

`plots/loss_vs_retain_kl.png` 和 PDF 为上下两个面板：上图四条固定样本损失，下图 Forget/Retain 两条表征 KL。上下图使用同一固定数据和同一次更新后的模型状态。固定样本损失先逐样本计算，再求均值；它与实际训练 batch 的聚合口径不同。

表征均取最后 RMSNorm 后、`lm_head` 前，两个 split 的位置分别为：

- **Forget**：预测整条答案中第一个敏感 token 的 `h[s-1]`。敏感标签为 `label >= 0` 且排除 EOS，跳过普通答案 token（`-200`）；只取最早敏感片段的第一个 token，不对多个片段取均值。该 token 也是 PALU 在 `FIRST_N >= 1` 或保留全部敏感 token 时的遗忘目标之一。
- **Retain**：没有敏感片段标签，继续使用预测答案首 token 的 `h[a-1]`。

当前模型与 Retain95 参考模型使用同一样本、同一位置和相同 teacher-forcing 前缀。Forget 表征能看到敏感 token 之前的普通答案内容，不能看到敏感 token 本身或其后文；不是读入该敏感 token 后的 `h[s]`。诊断使用：

```text
KL(softmax(h_Retain95 / T) || softmax(h_current / T))
```

这是隐藏维度 softmax 后的诊断量，不是词表输出概率 KL，也不能单独证明知识删除或决定最佳训练步。历史文件同时保存余弦距离。上图 Common-token KL 是 Forget 数据中普通答案 token 相对于初始 Full 模型的输出约束，方向为 `KL(current || Full)`；与 Retain95 表征 KL 区分使用。

Retain95 表征在训练前独立预计算并缓存，进程退出后才启动训练。训练诊断为 eval/no_grad，恢复 RNG 与模型状态；多卡各进程保持相同 forward 次数，填充样本不计入统计，只有主进程写历史。诊断不参与反向传播。

没有有效敏感 token（包括截断后只剩 EOS）的 Forget 样本会跳过，原因写入 `reference_cache.json` 的 `skipped`；固定数量不足时明确报错，`all` 表示所有有效样本。新缓存使用 schema 2，按 split 保存位置定义；旧答案首 token 缓存不能复用。正常启动会在新输出目录自动生成新缓存，无需删除旧实验。`probe_history` 记录两组位置定义，`probe_samples` 还记录逐样本预测位置索引。历史曲线重绘仍按旧定义标注，不会把旧数值改称敏感位置 KL，也不允许新旧位置混合聚合。

只保留四条训练曲线可设置 `DIAGNOSTICS=0`；纯性能测量使用 `DIAGNOSTICS=0 LOSS_HISTORY=0`。周期性诊断计入训练耗时，参考预计算单独计时。

## 输出、评估与多 GPU

默认采用 `{baseline}/{experiment}/{time}/{seed}` 四级路径，其中 baseline 根目录为 `artifacts/palu`，experiment 默认 `forget05_first_sensitive`，时间戳使用北京时间并包含微秒：

```text
artifacts/palu/
└── batch8_first_sensitive/             # EXPERIMENT：描述实验内容
    └── 20261013_120000_123456/         # 同一组 seed 共用时间戳
        ├── seed_0/
        ├── seed_1/
        ├── seed_2/
        ├── manifest.json
        ├── summary.csv / json / md
        ├── reference_cache.pt / json
        └── plots/
```

在 `run.sh` 设置 `OUTPUT_ROOT` 和 `EXPERIMENT` 即可使用新目录。一次设置 `SEEDS="0 1 2"` 会直接建立同组目录；分开启动时，相同实验名、相同配置且 seed 不重复的运行也会自动加入最新匹配时间戳：

```bash
EXPERIMENT=batch8_first_sensitive SEEDS="0" GPU_IDS=2 bash scripts/palu/run.sh
EXPERIMENT=batch8_first_sensitive SEEDS="1" GPU_IDS=7 bash scripts/palu/run.sh
EXPERIMENT=batch8_first_sensitive SEEDS="2" GPU_IDS=5 bash scripts/palu/run.sh
```

这三个命令的其余参数需一致。GPU 编号可以不同，GPU 型号、软件环境和训练拓扑仍会校验。匹配包含学习率、损失权重、微批次/梯度累计/进程数、模型路径/revision、数据与源码哈希、评估设置以及探针设置。改变 bash 的 seed 默认值不会单独改变分组。改变科学配置或重复同一 seed 会新建时间戳，不会覆盖已有结果；`NEW_GROUP=1`（CLI `--new-group`）可主动开启新组。

同组共享参考缓存；分开或并行启动时，文件锁保护 seed 注册、缓存预计算和汇总绘图。每个 driver 只训练本次请求的 seed，不会重新训练同组已有 seed。`manifest.json` 的 `launches` 保存每次启动的设置与源码/数据快照，各 run 记录自己的 GPU 编号。

汇总图保存在同一时间戳下的 `plots/`，每个点为该步各 seed 的均值，T 型误差棒为点态 95% Student-t 置信区间。`summary.json/md` 也报告指标均值和置信区间；单个 seed 时 CI 为空。尚未完成的 seed 不会伪造观测，图注显示实际 n。旧目录及曲线保留原样，不会自动移动或并入新组。

`OUTPUT_DIR` 或 CLI `--output` 仍支持旧的精确输出目录覆盖方式；使用新四级路径时请保持 `OUTPUT_DIR` 为空。`--dry-run` 独立生成计划目录，不占用实际实验组的 seed。

- `manifest.json`：参数、seed、完整命令、源码和输入哈希、可选参考预计算信息。
- `seed_<seed>/`：训练/评估日志、解析配置、运行时间和显存、`result.json`、`loss_history.jsonl/csv`。
- 开启诊断后：根目录 `reference_cache.pt/json`，各 seed 的 `probe_history.jsonl/csv`、`probe_samples.jsonl/csv` 和 `diagnostic_timing.json`。
- `summary.csv/json/md`：各 seed 的绝对指标，以及跨 seed 均值和样本标准差。
- `plots/`：PNG/PDF 曲线、`curve_statistics.csv`、`trend_intervals.csv`。最后一个文件仅标记相邻观测点的趋势相反区间，不是显著变点检验。

多 GPU 使用 Accelerate + DeepSpeed ZeRO-3，评估另起进程使用第一张可见 GPU。只启动一次 bash 文件，驱动会启动分布式 worker。每个 seed 从同一 Full checkpoint 开始，依次训练并评估。

默认 `KEEP_MODEL=0`：训练不保存中间 checkpoint，最终权重暂存用于评估；该 seed 成功完成并通过结果校验后删除其权重。日志、曲线、指标、tokenizer 和配置保留。训练或评估失败时保留权重以便排查。`KEEP_MODEL=1` 可保留最终权重。8B bf16 最终权重临时空间约 16 GB，另需初始/参考模型及其他缓存空间。

汇总检查 seed 配置除 seed 和输出路径外一致、步数一致、软件和设备环境一致。FQ 是 KS 检验 p-value，应结合 KS statistic、MU、EM 和流畅度解释。`--skip-eval` 仅用于训练/性能检查，不生成遗忘质量结论。

重建报告或重画图：

```bash
python scripts/palu/run_experiment.py --summarize artifacts/palu/<experiment>/<time>
python scripts/palu/plot_history.py artifacts/palu/<experiment>/<time>
```

旧双组实验目录请在原分支用原脚本读取，不要与新单组目录混合。

## 验证

```bash
python -m unittest discover -s tests/palu -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/palu/check_distributed_diagnostics.py
```

本分支验证范围见 [VALIDATION.md](VALIDATION.md)。
