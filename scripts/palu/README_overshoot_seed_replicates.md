# P336 / P421 / P386：三个新种子复验

本轮检验 TR 向高侧越过 Retain、FQ下降是否能在 F/R 损失降低、MU回升时跨种子重复出现。每个配置运行 seed 201/202/203，共9次，均从 TOFU **Full** 开始，不从基座或旧PALU模型续训。

| 配置 | lr | alpha | first_n | top_k | 主比较更新步 |
|---|---:|---:|---:|---:|---|
| P336 | 5e-5 | 5 | 3 | 5000 | 25→60（epoch4→末点） |
| P421 | 4e-5 | 5 | 5 | 1000 | 12→60（epoch2→末点） |
| P386 | 5e-5 | 5 | 4 | 10000 | 12→60（epoch2→末点） |

共同设置为：单GPU、batch8、累积4、原始microbatch均值、gamma1、mean目标、带梯度的mean、retain NLL、common KL、bf16、SDPA、paged AdamW、weight_decay0.01、warmup6 updates、线性学习率、60次更新。保存的源配置均为seed0，其余完整字段位于 `configs/analysis/overshoot_seed_replicates/`。历史world size缺失；单卡是本次明确冻结的执行协议，不能声称完整恢复了历史硬件环境。没有改成整步token等权。

## 启动

从包含本次提交的PALU仓库运行。当前服务器另有未提交实验代码，因此单独部署到 `PALU/launchers/overshoot_seed_replicates_20261019`；在该目录执行同样命令即可，环境和数据复用原仓库：

```bash
GPUS=0 bash scripts/palu/run_overshoot_seed_replicates.sh --dry-run
GPUS=0 bash scripts/palu/run_overshoot_seed_replicates.sh
```

默认在一张卡上顺序跑9次；若有三张空闲卡，可用 `GPUS=0,1,2` 并行运行三个独立单卡队列。这不改变每次训练的world size。每张卡默认要求空闲78 GiB；不足会退出，不会自动改变batch或停止其他任务。

```bash
GPUS=0,1,2 STUDY_DIR="$PWD/artifacts/overshoot_seed_replicates/my_run" \
  bash scripts/palu/run_overshoot_seed_replicates.sh
```

`PYTHON_BIN` 默认仓库 `.venv/bin/python`；`MODEL_PATH` 可显式指定同源TOFU Full快照，默认使用已验证的本地Full缓存；`RETAIN_LOGS` 可指定原Retain95评估，必须通过固定hash校验；`TOFU_DATA_PATH` 默认 `datasets/TOFU`（独立部署目录已链接原数据）。配置资产将原服务器绝对路径和输出名称替换为可移植字段，其余训练、数据和评估参数保留，原配置hash用于私有证据回溯。`SEEDS=201,202,203` 可替换为三个不同的正整数，启动后写入冻结清单。`--prepare-only` 只做CPU配置、数据、依赖与存储预检，不加载模型权重；`--analyze-only` 需指向已存在的 `STUDY_DIR`。

对相同 `STUDY_DIR` 再次启动，会核查代码/配置hash，并跳过已验证完成的run；遇到失败或不完整run会停下保留现场，不做未经验证的断点恢复。默认新启动会生成新时间戳目录。

## 损失、评估与空间

- 逐优化步：`loss_history.jsonl/csv` 记录未加权F/R/common-KL及加权三项、总loss。公式为 `gamma*F + alpha*R + alpha*common_KL`。这是train模式下参与更新的microbatch均值，不能与固定评估loss混为一谈。
- 每2个实际epoch即时评估，并额外评估初始和末点：step **0,12,25,37,50,60**。历史60-update预算对应1920条已消耗F/R配对，末点state.epoch=9.6；不偷偷延长到62/63步来凑10个完整epoch。
- 同时在固定全量F200/R3800上测三项loss，保存token加权均值、样本均值、逐样本loss和token数量。这些固定测量用于loss–FQ/MU散点；训练目标仍为原microbatch均值。
- `checkpoint-N/evals/TOFU_EVAL.json` 保留标准逐样本评估，`truth_ratios.csv` 显式导出每题TR、Retain TR、正确/错误候选几何平均分数和NLL。每点评估完成后重算KS/FQ及固定loss，写 `verification.json`。
- 评估使用当前内存模型并恢复随机数状态与训练模式，不落盘中间模型权重。清理仅在评估验证成功后处理当前run拥有的中间权重文件；保留 `checkpoint-N/evals/` 内的全部观测。最终模型留在各个 `配置/seed_N/` 根目录。
- 最终9份8B模型约135 GiB；启动会预留约5%余量及5 GiB结果空间。评估中断时保留日志及已完成结果；不保留可恢复的中间优化器checkpoint。
- `training_pairs.jsonl` 记录实际消耗的F/R配对，`prepared_pairs.jsonl` 区分预取但未消耗的样本。同一seed跨配置使用同一采样机制，分析前应核对实际配对记录；不能把seed效应分解为初始化、采样或GPU非确定性的独立效应。

## 预先固定的判定

P421/P386的历史峰值在epoch3，按两轮评估可能漏掉。因此主比较固定在上表区间，不根据新seed的FQ重新选峰。主要条件：锚点KS D≤0.15、末点D增加≥0.05、FQ下降≥100倍、MU增加≥0.01、固定测量的F和R分别下降、末点CDF以高TR侧偏移为主且中位数超过Retain、超过半数同题TR上升且平均logTR上升。common-KL完整报告，不把它暗中合入F或R。

`analysis/primary_seed_comparisons.csv` 逐项列出每个seed是否满足条件，`summary.json` 统计每组复现数/3，`all_epoch_points.csv` 保存原始逐点结果，`pairing_checks.json` 核验同一seed跨配置的实际F/R配对是否一致。`plots/` 给出各配置三种子的损失及指标曲线、FQ–F和MU–R散点等。FQ仍是标准KS p值，3个seed只构成初步稳定性估计；即使三次均复现，也不能证明已排除梯度冲突或真正删除了知识。
