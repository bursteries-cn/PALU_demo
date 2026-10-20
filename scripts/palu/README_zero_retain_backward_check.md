# 零权重 Retain 反向对照

该脚本只在独立进程中包装现有训练代码，不修改正式训练文件、不写模型 checkpoint、不运行评估，也不控制既有实验进程。只有 `raw_components` 中的 `retain_loss = model(**retain).loss` 在候选组被包进 `torch.no_grad()`。模型仍为 train 模式，保持 F → teacher → R 前向顺序、autocast、敏感 token、整步 token 分母及 `F + 0*R + 2*KL` 的算子顺序。

固定来源为本轮 seed103 的 `frozen/config.json` 和 `frozen/schedule.pt`：Llama-3.1-8B、学习率固定 `1e-5`、四卡每卡 batch1、累积8、32对 F/R、63步完整计划的前6步。重新从 Full 起点初始化模型、优化器与调度器。第1–3步作为 warmup，第4–6步作为计时区间。

## 2026-10-21 实测验证

服务器隔离目录 `artifacts/zero_retain_backward_check/20261021_043000` 以服务器提交 `cea38a3` 的复制源码运行，来源为 `artifacts/forget_threshold/20261020_165535/seed_103`。六步正确性比较已通过：52份初始/逐步状态与真实梯度分片记录逐位相同，三项损失、微批记录、FP32 master、Adam矩与step、scheduler、RNG也一致。baseline六步的配对、学习率、损失与grad norm与历史来源完全一致。该结果限于六步前缀，不宣称完整长训练已验证。

独立A/B/A六步重放（前3步预热）测得平均每完整更新53.4421 / 36.2200 / 53.3191秒。以两次原版平均为基准，耗时减少32.15%，速度1.474倍；原版前后漂移-0.23%。所有计时组的六步loss与初末状态也一致，并与正确性检查的共有初末状态一致。计时模式没有逐步gradient/state检查，不能从该模式单独推出每个中间更新相同；发布版已修正比较报告的范围描述，服务器原始证据保留原样。此小样本训练更新加速不能直接当作含评估的全流程加速。

复现这一已记录结果应使用该隔离目录的归档源码。后续正式入口增加了冻结策略和日志字段，不能将不同代码版本的产物直接混合做严格比较。新入口的A/B replay仍须使用上述未启用优化的原冻结计划；不要拿已经启用优化的plan作为baseline输入。

通过验证后，后续阈值实验可使用 `ZERO_WEIGHT_RETAIN_NO_GRAD=1`，见 [阈值实验说明](README_forget_until_threshold.md)。默认关闭、旧plan缺字段按关闭处理，R权重非零时始终正常反向。服务器主仓库未由本次测试更新，原实验未改动，没有启动正式训练。

## 启动

在隔离代码目录中执行，`SOURCE` 指向已完成的源实验，不在该目录写入结果。确认 GPU0–3 可用后运行：

```bash
SOURCE=/mnt/sda/cr/LLM_unlearning/PALU/artifacts/forget_threshold/20261020_165535/seed_103
CHECK_ROOT=/mnt/sda/cr/LLM_unlearning/PALU/artifacts/zero_retain_backward_check/独立时间戳
PYTHON_BIN=/mnt/sda/cr/LLM_unlearning/PALU/.venv/bin/python

CUDA_VISIBLE_DEVICES=0,1,2,3 "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/palu/zero_retain_backward_check.py --mode check --arm baseline \
  --study "$SOURCE" --output "$CHECK_ROOT/check_baseline"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/palu/zero_retain_backward_check.py --mode check --arm no_grad \
  --study "$SOURCE" --output "$CHECK_ROOT/check_no_grad"
"$PYTHON_BIN" scripts/palu/zero_retain_backward_check.py --mode compare \
  --baseline "$CHECK_ROOT/check_baseline" --candidate "$CHECK_ROOT/check_no_grad" \
  --output "$CHECK_ROOT/check_comparison.json"
```

比较进程返回0表示要求的记录逐位一致；返回2表示存在差异，不能当作等价通过。也支持两个同 arm 的重复实验比较，以检测基线自身的非确定性。缺记录、配置不一致、缺 Adam 状态、缺真实梯度会直接失败。

正确性通过后，以 `--mode time` 分别跑 baseline / no_grad / baseline（输出目录均不同），比较候选与前后两次 baseline，减少运行次序漂移的影响。time 模式仅在初始化与最终更新后进行完整状态检查，两次检查均在计时区间外；每步的计时从同样的 CUDA synchronize + barrier 后开始，结束于 CUDA synchronize，汇总采用每步四个 rank 墙钟耗时的最大值。保留正常训练日志开销，不包括模型加载、检查、评估与 checkpoint I/O。

## 对照内容

- 每个微批和完整更新的 F、R、common-KL、总损失、grad norm、LR、F/R配对身份。
- 第8次 backward 后、clipping/optimizer step 前，用 DeepSpeed `safe_get_local_grad` 读取全部可训练参数的真实本地 ZeRO3 累积梯度；不能用常为空的 `parameter.grad` 代替。四个 rank 的全部分片共同覆盖完整梯度。
- 初始与每步之后的 student/teacher 参数分片、buffer、FP32 master、Adam一二阶状态和step、调度器、引擎计数、Python/NumPy/Torch CPU/CUDA RNG、模块模式。初始与每步参数内容完全相同，意味着各次参数更新相同。
- 逐张量 SHA256、形状、dtype、numel、L2范数、最大绝对值。Adam状态遍历持久 FP32 master，不使用更新后可能为空的 `underlying.param_groups[*]['params']` 来推断状态。
- baseline 会将三项损失、grad norm、配对、LR 与来源 seed103 历史前6步对照。历史实验多执行了评估/probe，出现差异时需要单独归因，不以新的 A/B 相同掩盖来源轨迹偏移。

## 差异诊断

默认不存每步上百 GiB 的完整状态。若出现 hash 不一致，可重跑 baseline，指定例如 `--snapshot-step 1 --snapshot-write "$CHECK_ROOT/step1_tensors"`；候选重跑时对应使用 `--snapshot-step 1 --snapshot-compare "$CHECK_ROOT/step1_tensors"`。只在该步写出完整梯度和训练状态的原生分片，逐块比较，给出最大绝对差、L2差、相对L2差、RMS差、不同元素数量/比例。该选项可能需要百 GiB 级磁盘，应先检查余量。默认 hash 不一致时只报告未通过，不能用少数统计值相近替代全张量数值比较。

CPU测试只验证改动范围、微型模型的loss/gradient/Adam等价、梯度访问契约和二进制差异读取，不证明 Llama/ZeRO/GPU 等价。6步实验只能支持当前零 R 权重配置的短轨迹；R权重大于0时不能启用该修改。
