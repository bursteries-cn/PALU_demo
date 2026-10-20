# 先完成遗忘，保存可续训 checkpoint

后续新实验只训练 `F + 2×common-KL`。**每个 epoch 结束后首个完整更新处固定模型，重新评估全部 F200 的平均遗忘损失，首次 `<0.5` 时停止**，立即评估并保存模型、Adam 与各 rank 的 RNG。`0.4` 仅用于记录是否跌破下界，不是额外停止条件；越界也会停止，不继续训练或倒退选点。F200 平均仍按敏感 token 等权，不是训练过程中不同模型状态的 batch loss 平均。

**当前已经启动的每步检查实验保持原冻结计划，不改变运行进程、样本、停止条件或结果。** 此默认值只影响之后新准备的实验。历史 plan 缺少 `probe_interval` 字段时仍按每步检查、`≤0.5` 停止。**不进入保留阶段，不替用户选择 A/B/C。**

当前串行种子队列结束之前，不在服务器拉取或替换这些脚本：尚未开始的种子会读取磁盘上的代码，而已prepare的实验还会核对代码哈希。新版本仅在后续新实验开始前部署。

## 启动

在服务器 PALU 仓库中，使用已有 PALU 环境：

```bash
MODE=dry-run bash scripts/palu/run_forget_until_threshold.sh
FOUR_GPUS=0,1,2,3 bash scripts/palu/run_forget_until_threshold.sh
```

默认依次运行 `103 104 105 106`，每个种子复用同一组四卡。只跑某些种子：

```bash
SEEDS="104 105 106" FOUR_GPUS=0,1,2,3 \
  bash scripts/palu/run_forget_until_threshold.sh
```

默认 source 为 `artifacts/token_equal_seed103/20261017_192348`，新结果为 `artifacts/forget_threshold_epoch/时间戳/seed_数字`，与旧每步检查的 `forget_threshold/` 分开。可用 `SOURCE_STUDY`、`STUDY_DIR`、`PYTHON_BIN` 覆盖位置。`STUDY_DIR` 必须全新；现有结果不会被覆盖。每个种子启动前检查显存；任一失败后停止整个队列并保留观测及错误日志。

## 固定条件与阈值

- Llama-3.1-8B 从原始 Full 模型重新开始，每卡 batch1，累积8，四卡有效32对 F/R。
- 学习率恒定 `1e-5`、无 warmup；F、R、common 各自使用整步有效 token 分母。保留 PALU top5000、first3、mean target、梯度裁剪等原设置。
- common-KL 权重2始终生效，R 权重0。KL 是 Forget 非敏感位置上学生相对 Full 教师的 KL，并不是向 Retain 模型蒸馏。
- 默认 `PROBE_INTERVAL=epoch`：每经过200对曝光，在其后首个完整更新处，单独测固定 F200 的敏感 token 等权原始 F。它不同于随机微批 loss、乘权重的 objective，以及正确答案 NLL。
- 由于一整步包含32对，而一个 epoch 是200对，二者不总对齐。默认检查步为 `7,13,19,25,32,38,44,50,57,63`，对应 `1.12,2.08,3.04,4,5.12,6.08,7.04,8,9.12,10.08` 轮曝光。这是 epoch 边界之后首个完整更新，不声称精确停在200对边界；没有 pending gradients，最终状态可续训。
- 首次 `<0.5` 时停止；恰为0.5不会触发，恰为0.4不标记跌破下界。停止点可能已低于0.4，或跳过最佳FQ点，记录真实结果而不改选checkpoint。
- 默认最多63次完整更新，即2016对曝光、10.08轮。未达到阈值也评估并保存终点，明确标记 `budget_exhausted`，不声称阈值命中。
- 可通过 `THRESHOLD_LOW`、`THRESHOLD_HIGH`、`MAX_UPDATES`、`LEARNING_RATE`、`PROBE_INTERVAL` 在 prepare 前改设置。`PROBE_INTERVAL=update` 可复现每完整更新检查的协议，其默认比较为 `≤`；`THRESHOLD_COMPARISON=lt/le` 可显式指定。新 plan 冻结检查步和比较符号，训练只能读取该计划，不接受临时修改这些参数。

seed103 保留旧记录中的前60步配对，并按上一轮同一规则扩展到63步，是历史配对锚点。104–106 使用各自独立的 Torch F 排列生成器和 R 抽样生成器，R 从3800条中有放回抽样；不会仅改 seed 数字而继续重放103配对。冻结样本池相同，数据流、pair origin、样本覆盖量及哈希均留存。103与新种子的配对来源不同，因此不把四条轨迹笼统称为同一采样实现下的四次随机重复；104–106之间采用同一采样实现。

## 评估与留存

初始、每两轮曝光后首个完整更新、实际停止点均评估 FQ/MU、三项固定原始损失和逐样本 TR。最大预算下的常规评估步为 `0,13,25,38,50,63`；提前停止时只保留已到达的节点并额外评估停止点。第13/38步是第2/6轮边界之后首个完整更新，与旧实验在半累积状态评估第12/37步的时机不同。不会以未完成更新的梯度状态作为后续 A/B/C 的共同起点。

F200 probe 增加额外学生和教师前向计算。按epoch检查将63步预算内的阈值探针从63次减少到10次，减少了这部分工作量；实际总耗时还取决于停止步、FQ/MU评估、模型保存和I/O，不保证按同样比例加速。FQ/MU仍维持约每两轮一次，停止点另作完整评估。在线随机批损失、固定探针、固定评估损失有独立文件，后续分析应按各自口径比较。

- `frozen/`：配置、F/R 样本与完整计划配对、Retain 参考和文件哈希。
- `run/loss_history.jsonl`、阈值 probe 记录：逐步原始 F/R/common-KL 与实际系数、固定 F200 的停止依据。
- `run/epoch_history.jsonl`、`run/checkpoint-*/evals/TOFU_EVAL.json`：完整评估、各题候选概率与 TR。
- `run/stop_decision.json`：实际停止步、固定 F、是否落入0.4–0.5、是否越界或预算用尽。
- `run/checkpoint-停止步/`：最终可独立读取的模型。
- `run/resume_checkpoint/`：原生 DeepSpeed 训练状态与 `receipt.json`；保留模型分片、FP32 master、Adam 一二阶矩、scheduler、计数器及每 rank RNG，供后续同样四卡 world size、原模型与数据哈希校验后恢复。后续 A/B/C 控制器另行确定。
- `completion_validation.json`：停止规则、训练记录、评估、保存文件等最终校验；成功保存不等于已经完成GPU恢复等价性实验。
- `analysis/`：固定评估汇总、逐样本 TR 与 ECDF 数据；本次只准备遗忘终点，不绘制虚构的保留阶段。

评估成功后删除中间模型权重，最终模型与 resume 状态保留。**完整 Adam restart 比纯模型权重大得多**：BF16模型约16GB，含FP32 master和两项动量的训练状态约其7倍，每个种子还保存独立最终模型，合计约119GiB/seed、四个种子约477GiB。这是要求保留可续训 Adam 状态产生的最终占用，并非中间 checkpoint 清理失效。启动和 dry-run 都展示这项估计；可用 `SEEDS="104"` 先跑一个种子。脚本按实际 Full 权重字节数逐个检查可用磁盘，每个种子启动前至少需160GiB空闲；前面的最终结果持续保留，脚本不会自动删除它们。

实时输出：

```bash
tail -f artifacts/forget_threshold_epoch/实际时间戳/seed_103/logs/train.log
```

若训练中途失败，可读取已提交的评估并导出，不会假称完整实验通过：

```bash
python scripts/palu/forget_until_threshold.py --mode export \
  --study artifacts/forget_threshold_epoch/实际时间戳/seed_103
```
