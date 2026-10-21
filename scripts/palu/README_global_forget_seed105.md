# seed105：保留小学习率，恢复全体停止

本轮用于对照逐样本永久停止实验。学习率经确认保持 **5e-6**，不是5e-5；仅运行seed105的遗忘阶段，从原始Full模型重新开始。目标始终为 `F + 2 common-KL`，所有有敏感目标的Forget样本继续参与F项，直到全体停止，不再逐样本屏蔽F。

## 冻结条件

- 每个epoch边界后的第一个完整更新，固定模型评估全部F200；全数据集的敏感token加权平均F首次严格 `<0.5` 时结束，若 `<0.4` 则另记跌破下界。它不是在线batch损失平均，也不是200条样本均值的等权平均。
- 检查步为7、13、19、25、32、38、44、50、57、63；最多63次完整更新（10.08个200对曝光epoch）。未达阈值按预算结束并明确标记，停止后不自动转入保留阶段。
- 四卡，每卡batch1、累积8，有效batch32；整步token归一化、常数学习率、无warmup；PALU top5000、first3、mean target和其余源配置不变。R权重0，仅无梯度前向记录R损失。
- 沿用同一原始源实验和seed105独立采样器，从起点重新生成相同F/R配对。原始源为 `artifacts/token_equal_seed103/20261017_192348`，不是上一轮seed105终点。冻结计划、配对及哈希随结果保存。
- 专用入口固定上述种子、学习率、阈值、预算与停止粒度；不会受shell中遗留的同名变量影响。GPU、Python和路径变量仍可配置。其他实验请使用通用入口。

## 启动和输出

在服务器PALU目录下，先查看计划，再正式启动：

```bash
MODE=dry-run bash scripts/palu/run_global_forget_seed105.sh
FOUR_GPUS=0,1,2,3 bash scripts/palu/run_global_forget_seed105.sh
```

如未激活PALU环境，可在命令前加 `PYTHON_BIN=.venv/bin/python`。默认结果目录为 `artifacts/global_forget_seed105/<时间戳>/seed_105`，日志为该目录下的 `logs/train.log`。可用 `STUDY_DIR` 指定全新目录，用 `SOURCE_STUDY` 指定同一历史源目录的实际位置。dry-run不加载模型、不创建结果目录、不启动GPU训练。

F200上的F和common-KL每轮评估；FQ及逐样本TR在初始、每两轮及实际终点评估，MU仅在终点评估一次。逐更新记录原始F/R/common-KL，保存逐样本探针和TR、CDF分析数据。评估验证后删除中间模型权重，保留最终模型、原生优化器/RNG状态和全部测量结果。启动前仍检查至少160GiB可用磁盘，单seed最终模型加可续训状态约100–120GiB。

训练结束后可单独绘制损失、指标与TR曲线（将时间戳替换为实际目录；需要matplotlib）：

```bash
python scripts/palu/plot_sequential_phases.py \
  --study artifacts/global_forget_seed105/实际时间戳/seed_105 \
  --output artifacts/global_forget_seed105/实际时间戳/seed_105/figures
```

## 解释边界

本轮相对上一轮seed105只改变停止策略，适合比较逐样本屏蔽与保持全部F项的轨迹。共同更新步上的对照比两个不同停止步的终点更直接；终点训练量可能不同。全局F阈值不保证每条样本处于区间，也不代表TR已经匹配或越过Retain。首次F低于0.5就停止，且每两轮记录TR，可能截断或漏掉过冲过程；本轮据实报告是否观察到，不预设其必然发生。
