# seed103：四卡整步 token 归一化与第 6 步分叉

GitHub 提供代码与测试；本实验的冻结配对、原配置和历史观测文件使用服务器已有的 `configs/analysis/layout_loss_seed103/`，不随公开代码发布。运行准备时会逐文件验证这些本地输入的 hash。若采用另一个干净 checkout，应先将这个目录从原 PALU checkout 复制或链接到同名位置；已有 PALU 目录可直接复用。该目录缺失时，脚本无法重建原 seed103 的实际配对，不应重新采样代替。

在服务器 PALU 仓库执行：

```bash
# 只核对协议，不加载模型，也不启动 GPU 任务。
bash scripts/palu/run_seed103_branches.sh --dry-run

# 将编号换为四张可用卡。公共前缀及三支顺序执行，不需要额外 GPU。
FOUR_GPUS=2,3,4,5 bash scripts/palu/run_seed103_branches.sh
```

默认 Python 是仓库 `.venv/bin/python`，可设 `PYTHON_BIN`。设 `MODEL_PATH`、`RETAIN_LOGS` 可覆盖服务器路径，但冻结数据和 reference 的内容 hash 必须与 seed103 相同。默认输出 `artifacts/seed103_branches/<北京时间>/`；建议显式设 `STUDY_DIR`，便于查看与重新分析。

本实验固定 **4 卡，每卡 batch1 × 累积8，有效 batch32**。F、R、common KL 分别以整个32对样本更新窗口的对应有效 token 总数归一化，补偿 DeepSpeed 的梯度累积与跨卡平均。训练 padding 保留最初四卡的每条样本独立 padding；没有同时改成整步 padding。

其余沿用 Llama-3.1 seed103：Full snapshot `1a5c5b1a557f8c99bdadecd5168ebd03f640b00e`，Forget05/Retain95=200/3800，α=2、top_k=5000、first_n=3、当前 mean target 参与梯度、Retain NLL、common KL(current||Full)，LR=2e-5、paged_adamw_32bit、weight decay=.01、clip=1、BF16/SDPA、gradient checkpointing、四卡 ZeRO3 FP32 master（当前配置的offload device=none）、linear scheduler、warmup6、每条路径总计60次完整更新。重放原 seed103 实际消费的1920对 F/R 样本，顺序与配对不变，不重新抽样。总预算为9.6次实际 Forget 遍历，名义配置 `num_train_epochs=10` 不等于10个完整 epoch。

公共前缀先使用 γ=1 完成 **6次完整更新、192对样本**，在无待累积梯度的边界保存一次完整 DeepSpeed restart。随后三支各新建一个四卡进程组，加载同一份模型、Adam moments、FP32 master、scheduler、完整更新计数与各rank RNG。恢复后逐rank核对状态指纹，设置分支 γ，然后从第7次更新继续：

| 目录 | 第1至6次更新 | 第7至60次更新 | 研究问题 |
|---|---|---|---|
| `continue` | F+2R+2KL | F+2R+2KL | 持续联合训练基线 |
| `repair` | F+2R+2KL | 2R+2KL | 关闭敏感遗忘项后效用能否恢复 |
| `weak` | F+2R+2KL | 0.1F+2R+2KL | 较弱遗忘约束能否抑制恢复时的反弹 |

γ=0 分支仍对 Forget 的非敏感位置施加 common KL，因此属于关闭敏感遗忘项的恢复对照。三支不重置优化器、LR、warmup、调度器或后续配对序列。公共前缀只训练一次，实际总执行6+3×54=168个更新；每条比较路径仍然是60个更新。

**2026-10-17 协议更新：默认每两个实际 epoch 评估一次，采用四卡按样本分工。** 第6次完整更新、192对样本后分叉的训练设置保持不变；新的中间评估不在epoch1执行。实际评估消费对数为 `0,400,800,1200,1600,1920`，对应权重step `0,12,25,37,50,60`。总预算仍是9.6个实际epoch，最后不足两轮的尾段也必须评估。60次更新的训练损失仍全部记录。可用 `EVAL_EVERY_EPOCHS` 和 `EVAL_BACKEND=legacy|sharded` 显式设置新实验，准备阶段写入冻结协议；已有目录协议不一致时拒绝覆盖，旧实验分析仍读取其原有节奏。

新版固定loss保持batch1，各卡处理不同样本；TOFU将原始batch32整个分配给各卡，保持原始成员、padding和精度。每个评估阶段不足四份时使用占位计算维持ZeRO前向次数一致，占位结果丢弃。合并后按原样本顺序恢复全量逐样本结果，使用原生公式计算FQ/KS D/MU，不对四份p值或MU取平均。生成采用同步终止；针对安装版Transformers 4.45的DynamicCache行为，已结束卡的无效陪同前向结束后回滚该次cache追加，实际输出和未完成卡的解码保持原生路径。

每次执行：暂停训练 → 保存临时权重 → 原地评估 → 核对全rank训练状态和完整逐样本结果 → 提交结果 → 删除该中间权重 → 继续下一个微批。初始状态只评估一次，其结果复用于三支；最终step60模型保留。待累积梯度、Adam、FP32 master、RNG不重置。失败时保留现场。评估频率降低会减少早期轨迹的观测密度，不能再从新版推断epoch1附近的细粒度变化。

这次协议更新不能通过旧的兼容性恢复入口应用到已运行半程的目录。请保留旧观测、checkpoint及完整fork；新评估验证使用独立目录，不改写旧实验记录，也不自动续训。

固定测量口径：全量F200/R3800、batch1、token加权F/R/KL，同时保存样本宏平均和逐样本损失；FQ、KS D、MU沿用冻结Retain95 reference及原TOFU逐样本评估。评估中的canonical total恒为 **F+2R+2KL**，便于跨支比较；在线训练的`loss_total`按各步实际γ计算，`gamma_used`、`alpha_used`及canonical total另外记录。因此恢复支的总训练loss会因目标项被关闭而发生定义变化，应优先比较raw F/R/KL，不能用不同定义的总loss判断“同损失”。

`analysis/` 自动保存训练损失、固定损失、FQ/MU/KS D折线，以及FQ-F、MU-R、D-F二维散点与轨迹，均提供PNG/PDF；另保存所有状态CSV、F/R共同接近的跨支点对和验证汇总。绘图复用`plot_seed_epoch.py`的坐标轴、标签和导出逻辑，三支以不同策略绘制，不伪作三个种子，不计算跨策略均值或标准差。零FQ完整留在结果中，log轴不以epsilon替代零。

保留内容包括三支各自的`checkpoint-60`最终模型、`loss_history.jsonl`、`training_pairs.jsonl`、`epoch_history.jsonl`、`fixed_loss_samples.jsonl`、全部原始TOFU逐样本结果、配置/数据/代码来源hash、逐次评估及恢复验证记录、图表。公共restart tensor位于`fork_weights/`，审计receipt位于`fork/receipt.json`；三支完成、所有最终模型和评估结果通过汇总校验、图表生成成功后，才删除`fork_weights/`。原始receipt保持不变，另写`fork/cleanup_receipt.json`记录删除量和验证对象。

**磁盘峰值主要来自一次完整优化器分叉状态。** 以BF16 safetensors实际字节数M动态估计：共享restart约7M（BF16模型+FP32 master+两个FP32 moment），三个最终模型3M，临时权重及时删除后峰值约10M。Llama3.1-8B的最终模型合计约44.9 GiB，新增权重与restart峰值约149.6 GiB；预检再加10%及8 GiB结果/序列化余量，约172.5 GiB空闲。Full起点已有权重不复制。每个阶段按目前空闲空间和剩余final数重算，不把已经写入的restart再重复计为空闲空间需求。若实际DeepSpeed格式额外增大，空间不足会停止并保留记录。

显存预检仅针对四卡，每卡默认至少40 GiB空闲，可用`MIN_FREE_GIB`调整；该门槛是启动保护，不是峰值显存保证。不会等待、占用或终止其他任务。这里不再套用旧单卡78 GiB门槛。

```bash
# 仅CPU准备，随后沿用相同目录正式执行。
STUDY_DIR=artifacts/seed103_branches/my_run \
  bash scripts/palu/run_seed103_branches.sh --prepare-only

STUDY_DIR=artifacts/seed103_branches/my_run FOUR_GPUS=2,3,4,5 \
  bash scripts/palu/run_seed103_branches.sh

# 完成后重建汇总与图；完整验证通过才清理尚存的公共restart tensor。
STUDY_DIR=artifacts/seed103_branches/my_run \
  bash scripts/palu/run_seed103_branches.sh --analyze-only
```

已完成的公共前缀及分支可复用；每次载入分叉仍检查restart与receipt。不支持任意中断点的半步继续训练：已经恢复fork并进入实际分支训练的partial run会被拒绝，应保留失败目录。不要手动把incomplete状态改成new或completed。

如果公共前缀已成功保存，而首次分支在读取DeepSpeed原生checkpoint时因`weights_only=True`默认值失败，可显式复用第6步fork，恢复后从第7次完整更新继续，前6步不重跑。恢复命令必须显式设置原`STUDY_DIR`。`FOUR_GPUS`建议沿用原实验物理设备，可查原`logs/continue_gpu_before.log`和分支manifest的`cuda_visible_devices`；脚本不会自动猜测GPU编号：

```bash
# 先只做CPU恢复准备，不加载模型、不启动GPU。
STUDY_DIR=artifacts/seed103_branches/原实验目录 \
  bash scripts/palu/run_seed103_branches.sh --resume-from-fork --prepare-only

# 核验成功后从同一fork运行三支，前6步不重跑。
STUDY_DIR=artifacts/seed103_branches/原实验目录 FOUR_GPUS=0,1,2,3 \
  bash scripts/palu/run_seed103_branches.sh --resume-from-fork
```

恢复前逐项校验5个冻结输入文件、seed103来源、prefix已完整完成6步/192对且无待累积梯度、fork全部文件hash、旧代码snapshot与清单。兼容补丁只允许改变`seed103_branch_training.py`、本控制脚本、本启动脚本和本README；其余损失、数据、padding、评估、优化器设置代码只要变化就拒绝复用。旧代码清单及完整snapshot保存在`compatibility_audit/<时间>/`，再更新这4个文件的snapshot及hash，receipt完整记录前后版本。冻结数据、公共前缀和fork文件保持原样。

自动归档仅限**尚未成功恢复fork、completed_updates=0、目录只有manifest/resolved_config且没有训练或评估记录**的失败初始化目录：整体移入`failed_load_attempts/`，不删除；对应的训练报错、磁盘预检和GPU预检日志一并复制到归档目录的`logs/`，原日志也保留，避免下一次启动tee覆盖后丢失失败证据。存在`fork_restore_receipt.json`、任何实际训练记录或中间checkpoint的partial run会拒绝自动处理。GPU正式训练仍由用户运行启动命令；`--resume-from-fork --prepare-only`始终只做CPU校验与归档。

这是单一seed、单一切换时点的配对干预，能检验当前条件下恢复阶段的效果。MU回升时还应联合检查KS D、逐题TR与正确/错误答案损失；单独FQ变大或变小均不能证明知识删除或恢复。新版完整三分支训练仍待实际运行；CPU协议检查或独立checkpoint评估通过，不等于三分支训练已经完成。

### 同 checkpoint 评估与计时验证

```bash
STUDY_DIR=/absolute/path/to/source_study \
CHECKPOINT=/absolute/path/to/source_study/continue/checkpoint-43 \
BENCHMARK_DIR=/absolute/path/to/new_validation FOUR_GPUS=0,1,2,3 \
  bash scripts/palu/run_evaluation_benchmark.sh
```

两个后端顺序使用相同四卡，各自新建进程、读取同一checkpoint，Full teacher与原冻结输入相同；不训练、不恢复优化器、不删除或修改源模型。`comparison.json`逐项比较固定loss、所有原生TOFU逐样本数值与生成文本，并报告FQ/MU/KS D、固定loss/TOFU/总耗时及各rank实际工作量。预设loss容差atol=rtol=1e-6、FQ/MU/D绝对容差1e-12；同时报告精确相等数量及最大误差。计时排除模型加载与状态hash，另报告全rank同步端到端评估时间。每后端一次计时不能估计稳定加速比的方差。

`SMOKE=1`仅使用F7/R11及每个原生评估集前65条，供同步和尾批检查；该结果不能代替全量F200/R3800与原TOFU协议的验证。
