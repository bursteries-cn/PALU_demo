# seed103 小学习率：先遗忘，再保留

本轮是单条连续轨迹，目的在于观察 TR 分布接近、越过或返回 Retain95 参考的过程。小学习率不保证出现过冲，也不保证每两轮评估能捕获 FQ 的峰值。脚本完成与科学实验完成分别记录。

默认配置（均可在启动前查看）：

| 项目 | 设置 |
|---|---|
| 模型/种子 | Llama-3.1-8B TOFU Full / seed103 |
| 训练 | 四卡，每卡 batch1，累积8；有效 batch32；整步分别按 F/R/common 有效 token 总数归一化 |
| 学习率 | **恒定 1e-5，warmup=0**；这是对旧 linear/warmup6 的明确变更 |
| 第一阶段 | 仅 PALU 遗忘项：F；目标 10 个 F200 曝光轮次，向上取整为63次完整更新/2016条F |
| 第二阶段 | 2R + 2common-KL；目标 10 个200对曝光轮次，63次完整更新/2016条R抽样 |
| common-KL | 作用于 Forget 输入的非敏感位置，相对原始 Full teacher；不是 Retain 模型蒸馏 |
| 阶段切换 | 第63次完整更新后切换，第64次开始新目标；继承 Adam 状态，不重启训练 |
| 其他 PALU 参数 | top_k5000、first_n3、mean target（保持原梯度路径）；BF16/SDPA、原优化器、clip1、weight_decay0.01 |
| 评估 | 初始、各阶段每2个曝光轮次、阶段终点；固定 F200/R3800 损失和原生 TOFU FQ/MU/逐题TR |
| 存储 | 评估落盘并校验后删除中间模型权重；保留最终模型、逐步损失、全部逐样本评估、图表、配置和代码快照 |

**轮次口径**：默认两阶段的一轮均为200对曝光。保留阶段的2016次R抽样是从3800条中有放回抽样，并非遍历Retain十次；实际唯一R覆盖数写入 `experiment.json`。设置 `RETAIN_EPOCH_SIZE=3800` 可改为每轮3800次抽样，但仍是有放回抽样，不能称作无遗漏遍历。

默认评估完整更新编号为 `0,12,25,37,50,62,63,75,88,100,113,125,126`。名义epoch边界若落在梯度累积中，评估当前已完成更新的权重，保存并恢复未提交梯度；日志同时记录 `consumed_pairs / updated_pairs / pending_pairs`。两个阶段分别以各自曝光量计数，阶段终点再补一次完整更新后的评估。因此10轮附近有额外短间隔评估。

采样来源为已完成 token 等权实验的 `frozen/`。前60次更新、1920对F/R及其配对逐条保持，逐窗口hash校验；后续先补齐第10次Forget遍历缺失的80条，再由独立 `random.Random(103)` 生成新的Forget排列和Retain有放回抽样。扩展部分是新冻结序列，不声称为历史seed103未记录的延续。种子相同本身不意味着数据流相同。

## 启动

在服务器PALU环境、仓库根目录中执行：

```bash
MODE=dry-run bash scripts/palu/run_sequential_forget_retain.sh
FOUR_GPUS=0,1,2,3 bash scripts/palu/run_sequential_forget_retain.sh
```

默认来源为仓库内 `artifacts/token_equal_seed103/20261017_192348`，可用 `SOURCE_STUDY` 指定同源冻结实验。输出自动新建于 `artifacts/sequential_forget_retain/时间戳/`，已有目录不会覆盖。`PYTHON_BIN` 默认使用激活环境的 `python`；也可显式指定。启动前检查四卡空闲显存（默认40GiB/卡，可用 `MIN_FREE_GIB` 配置），不停止其他任务。

例如调整阶段预算，或去掉第二阶段common-KL：

```bash
FORGET_EPOCHS=20 RETAIN_EPOCHS=10 LEARNING_RATE=1e-5 \
  COMMON_KL=0 FOUR_GPUS=0,1,2,3 \
  bash scripts/palu/run_sequential_forget_retain.sh
```

查看输出：

```bash
tail -f artifacts/sequential_forget_retain/时间戳/logs/train.log
python scripts/palu/sequential_forget_retain.py --mode plot \
  --study artifacts/sequential_forget_retain/时间戳
```

结束后自动校验并绘图；中途 `--mode plot` 只读取已落盘结果。记录每次更新的三项**原始**损失及实际加权目标；即使某项权重为0也保留其数值，不能把加权零损失当作该能力不再变化。固定全集评估与在线更新前损失分别展示。

分析应同时看FQ、KS距离及高/低TR两侧CDF差、TR分位数、MU与三项原始损失。FQ是KS检验p值，最大化FQ不等于证明知识删除。R恢复阶段也可能恢复目标答案偏好；Adam动量使刚切换时的更新仍受上一阶段影响。此单轨迹实验能展示动态，不能单独区分学习率、训练顺序、训练预算及common-KL的因果贡献。
