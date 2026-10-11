# PALU 全词表均值梯度消融

从远端最新分支 `origin/feature/npo-representation-batching` 的 `16cb23de5c168e2b1efcb30665c5bc78ebc997c8` 创建，实验分支为 `experiment/palu-mean-gradient-ablation`。

这个实验只改变当前模型全词表均值是否参与反向传播，比较两条优化路径：

| 实验组 | Hydra 配置 | 含义 |
|---|---|---|
| `full_mean` | `trainer.method_args.detach_mean_target=false` | 默认，保留官方行为 |
| `stop_mean` | `trainer.method_args.detach_mean_target=true` | 对同一个全词表均值执行 `detach()` |

两组的均值仍随当前模型变化。在相同 logits 上，前向损失完全相同；参数更新后，两组的模型、均值和损失可以逐渐分化。`stop_mean` 不是固定参考模型均值，也不是 top-K 均值。

## 一键启动与集中设置参数

在服务器项目目录激活 PALU 环境后：

```bash
bash scripts/palu_mean_ablation/run.sh
```

参数集中在 `run.sh` 开头，默认四卡、seed 0/1/2、学习率 2e-5、alpha=0.2、gamma=1、top_k=5000、first_n=3、10 epoch。修改文件中的默认值即可；环境变量也可以临时覆盖：

```bash
LR=1e-5 ALPHA=0.3 bash scripts/palu_mean_ablation/run.sh
```

默认开启损失记录及 Retain-only 表征诊断；不保留成功完成的最终模型权重。脚本复用同一个配对驱动，两组共享全部参数。终端实时显示子进程日志，完整内容仍写入各组 `train.log`。

需要检查的路径是 `MODEL_PATH`、`RETAIN_MODEL_PATH`、`RETAIN_LOGS`、`DATASET_PATH`、`CLASSIFIER_MODEL`。默认 Retain 权重是公开的 `open-unlearning/tofu_Llama-3.1-8B-Instruct_retain95`，也可改成已下载的本地模型目录。`RETAIN_LOGS` 是原有 FQ 参考 JSON，两者需对应同一个参考 checkpoint 和 split；评估 JSON 无法代替模型权重。服务器数据路径默认保留实际拼写 `/mnt/sda/cr/LLM_unlearning/datset/TOFU`。

不加载模型的启动检查，以及多卡短跑：

```bash
bash scripts/palu_mean_ablation/run.sh --dry-run
SEEDS="0" MAX_STEPS=2 PROBE_EVERY=1 bash scripts/palu_mean_ablation/run.sh --skip-eval
```

短跑只有一个 seed，不能估计跨 seed 置信区间，也不测量遗忘质量。正式图默认来自三个 seed。运行中的代码和启动脚本纳入 manifest 哈希，实验过程中不要编辑它们；环境变量覆盖可用于下一次实验。

## 四条训练损失与 T 型误差棒

`loss_curves.png/pdf` 含两个面板，分别显示 full_mean、stop_mean，每个面板四条折线：

| 曲线 | 记录字段 | 定义 |
|---|---|---|
| Total | `loss_total` | 下列三项之和 |
| Forget | `loss_forget` | gamma × 敏感 token 遗忘损失 |
| Retain NLL | `loss_retain_nll` | alpha × Retain 数据答案 NLL |
| Common-token KL | `loss_common_kl` | alpha × Forget 数据普通答案 token 的 KL(current ∥ 初始 Full) |

训练损失每个 optimizer step 写入 `loss_history.jsonl/csv`；先平均实际梯度累积窗口内各 microbatch 的原始目标，再汇总各 GPU，不改变原训练目标或梯度。原始未加权的三项也保存在日志中。损失值发生在本次更新前；横轴编号是它对应的更新编号。

同一个 step 汇总两个实验组都已有记录的相同 seed，绘制均值及逐点 95% Student-t 置信区间：`mean ± t(0.975, n-1) × sample_std / sqrt(n)`，每个可估计的点均有带端帽的 T 型误差棒。图和 `plots/curve_statistics.csv` 记录实际 n 与使用的 seed。不会插值缺失 step，也不会把 microbatch 当成独立 seed。n=1 时只画点和折线，不伪造置信区间。n=3 的区间可能很宽；它描述随机种子的变异，采用 t 区间的通常假设，不是整条曲线的同时置信带。

## Retain-only 最后表征 KL

新增诊断参考是只用 Retain 数据训练的 checkpoint，与 PALU 原损失的冻结 Full 参考分开。默认固定抽取 32 条 Forget、32 条 Retain 训练样本，在 step 0、每 5 个 optimizer step、最终更新后测量；样本及顺序跨两个实验组和所有 seed 共享。调整 `PROBE_EVERY`、`PROBE_FORGET_SAMPLES`、`PROBE_RETAIN_SAMPLES`；样本数可以设置为 `all`。默认只表示固定训练子集，不表示全训练集。

若答案首 token 的输入位置为 a，提取最后一层 Transformer 经最终 RMSNorm 后、进入 `lm_head` 前的 `h[a-1]`，即预测首 token 的状态。定位包含 PALU 的 -200 普通答案标签，不能用第一个敏感标签代替答案首 token；无有效答案、缺少前驱、截断等样本会跳过并记录原因。使用与训练相同的 token IDs、attention mask 和模板。参考预计算直接调用 Llama decoder，当前模型通过真实包装后的 LM 前向与 pre-head hook 提取，避免绕过 ZeRO-3。

对隐藏维度作 softmax：`q_R=softmax(h_R/T)`、`q_U=softmax(h_U/T)`，记录 `KL(q_R ∥ q_U)`，隐藏维度求和、有效样本求平均，FP32 计算、自然对数单位 nats，默认固定 T=1。它是表征坐标归一化后的诊断，不是词表概率 KL；随温度、尺度和坐标系变化，不能单独证明删除。额外记录余弦距离辅助核对。当前支持同架构、同词表的 Llama；参考 checkpoint 还应具有相同预训练来源。

启动前另开进程加载 Retain-only 模型，预计算后退出；训练期间仅保留小型 CPU 参考缓存，无第三个常驻大模型。`reference_cache.pt/json` 保存样本、位置、参考版本/本地权重哈希和数据配置；两组每次训练前验证缓存 SHA-256，Trainer 再核对选定样本与 Tokenizer。缺少 checkpoint 或不兼容的缓存会明确报错，不会退回 Full 参考。

在同一诊断时刻，以 eval/no_grad 在固定样本重新计算四项损失与表征 KL，保存 `probe_history.jsonl/csv` 和逐样本 `probe_samples.jsonl/csv`。固定样本损失是逐样本目标的平均，和实际训练的 microbatch 平均分别标记。`loss_vs_retain_kl.png/pdf` 上排四条固定样本损失，下排 Forget/Retain 两条表征 KL，共享 step 横轴，误差棒仍来自跨 seed 的同一口径。`plots/trend_intervals.csv` 保存相邻测量区间的变化量和 loss 下降/KL 上升的候选区间；不把它当显著性检验或精确变点。间隔为 5 时只能定位到区间。KL 接近 Retain 模型不保证 FQ 增大。

诊断保存并恢复随机数状态及模型模式，不更新参数、优化器或 scheduler。所有 GPU 参与相同次数的前向，重复补齐样本的统计权重为零，只有主进程写日志。新增诊断会增加耗时；完整训练入口耗时包含它，参考预计算作为共享阶段另记在 manifest；`diagnostic_timing.json` 记录各次诊断的最大每卡耗时，并区分任何晚于 Trainer runtime 统计的异常终止补测。不建议用简单相减推断纯训练速度。性能实验可用：

```bash
DIAGNOSTICS=0 LOSS_HISTORY=0 bash scripts/palu_mean_ablation/run.sh
```

每完成一个 seed 的配对实验，自动重建 PNG/PDF 和曲线统计；实验过程中或结束后也可手动重画：

```bash
python scripts/palu_mean_ablation/plot_history.py artifacts/palu_mean_ablation/<本次输出目录>
```

旧实验只保存总损失时无法补出历史分项或 KL，需重新运行。

## 先验证梯度事实

在仓库根目录、项目 Python 环境中运行：

```bash
python scripts/palu_mean_ablation/probe_gradients.py \
  --output artifacts/palu_mean_ablation/gradient_probe.json
```

探针调用真实 `PALU.compute_loss`，隔离敏感位置的遗忘项，令保留项权重为零。默认 logits 为 `[5, 3, 0, -2, -1]`，K=2，应得到：

| 实验组 | loss | logits 梯度 |
|---|---:|---|
| full_mean | 10 | `[2.8, 0.8, -1.2, -1.2, -1.2]` |
| stop_mean | 10 | `[4, 2, 0, 0, 0]` |

还可以在 Llama-3.1 词表规模上核验解析公式：

```bash
python scripts/palu_mean_ablation/probe_gradients.py \
  --vocab-size 128256 --top-k 5000 \
  --output artifacts/palu_mean_ablation/gradient_probe_large.json
```

这里使用随机 logits，不加载 8B 模型。它验证梯度支持范围，不测量遗忘效果或训练速度。

## 配对训练与独立评估

沿用 `requirements.txt` 的依赖版本，特别是 `torch==2.4.1`、`transformers==4.45.1` 和 `accelerate==0.34.2`。正式实验需要 CUDA、可加载的原始 TOFU 模型、TOFU 评估数据及配套 Retain 模型的 `TOFU_EVAL.json`。不能用 `TOFU_SUMMARY.json` 替代样本级参考结果。

默认设置：forget05/holdout05/retain95、K=5000、每个敏感跨度 first_n=3、lr=2e-5、alpha=0.2、gamma=1、10 epochs、SDPA、梯度检查点。默认使用 GPU 0,1,2,3，每卡 batch=1、累积=8，有效 batch=32；独立评估使用第一张卡，batch=1。两组都使用 NLL 保留项与原有普通位置约束。默认 seed 为 42、43、44。

默认不保留最终权重。独立评估仍需要临时 checkpoint：每对实验组均训练、评估成功并通过配对校验后，删除两组权重和权重索引，保留配置、tokenizer、训练状态、日志和全部指标。失败或未完成的一对保留权重供排错。要保留权重，显式添加 `--keep-model`。`--skip-eval` 的训练检查也会在配对校验后清理权重。

先生成计划，检查模型和参考路径：

```bash
python scripts/palu_mean_ablation/run_comparison.py \
  --gpu 0,1,2,3 --seeds 42 43 44 \
  --model-path /absolute/path/to/tofu_full_model \
  --retain-logs /absolute/path/to/retain95/TOFU_EVAL.json \
  --output artifacts/palu_mean_ablation/plan \
  --dry-run
```

正式运行用新的输出目录：

```bash
python scripts/palu_mean_ablation/run_comparison.py \
  --gpu 0,1,2,3 --seeds 42 43 44 \
  --model-path /absolute/path/to/tofu_full_model \
  --retain-logs /absolute/path/to/retain95/TOFU_EVAL.json \
  --output artifacts/palu_mean_ablation/tofu_forget05
```

模型路径可以是本地完整 checkpoint 或 Hub id；使用 Hub id 时，可用 `--model-revision <commit>` 固定模型和 tokenizer 的版本。Llama-2 chat 可通过 `--model Llama-2-7b-chat-hf` 选择，参考日志必须与模型和 split 匹配。

离线评估支持 `--eval-dataset /absolute/path/to/TOFU` 和 `--classifier-model /absolute/path/to/fluency_classifier`。后者只指定已有评估器的流畅度分类器。共享训练参数可用重复的 `--override` 传入，例如：

```bash
--override trainer.args.optim=adamw_torch
```

多卡使用仓库的 Accelerate + DeepSpeed ZeRO-3 配置，分片训练模型、参考模型和优化器状态，需要 `deepspeed==0.15.4`。只运行一次这个驱动程序，它负责启动各卡进程；两组依次执行，不在同一组显卡上同时训练。可用 `--main-process-port 29514` 避免与其他分布式任务的端口冲突。单卡用 `--gpu 0`，双卡用 `--gpu 0,1`。

未指定 `--grad-accum` 时，脚本自动选择累积步数，使 `GPU 数 × 每卡 batch × 累积步数 = 32`；无法整除会要求显式指定。可通过 `--batch-size` 和 `--grad-accum` 改变预算，但两组必须共享相同设置。需要比较固定优化步数时，用 `--max-steps 100`。

评估另起单卡进程，因此不受仓库多进程训练跳过自定义评估的限制。临时磁盘空间仍需容纳一对最终权重；8B bf16 权重两份约 32 GB，不含初始模型缓存和其他输出。ZeRO-3 保存完整权重还涉及聚合与写盘，参考模型初始化在分片前也有临时显存开销；多卡并不保证任意显存配置均能运行。

## 输出与实验有效性

每个实验组使用独立 Python 进程，从相同 checkpoint 开始；训练 seed 与数据采样 seed 一致。不同 seed 交替组别顺序，降低固定运行顺序的影响。训练时关闭评估，保存最终模型后另起进程运行 TOFU 评估。

脚本保存：

- `manifest.json`：精确命令、共同设置、代码与可见本地输入文件的 SHA-256。
- `gradient_probe.json`：真实遗忘损失的梯度验证。
- `seed_<seed>/<variant>/resolved_config.json`、`trainer_state.json`、训练日志和权重清理记录 `result.json`；默认成功完成后不保留最终模型。
- `telemetry.json`：实际依赖版本、各 GPU 名称、CUDA 分配/保留显存峰值和训练入口耗时；多卡同时保留 `telemetry.rank<N>.json`，避免进程互相覆盖。
- `evals/TOFU_EVAL.json`、`TOFU_SUMMARY.json`、评估来源记录。
- `comparison.json`、`comparison.csv`、`comparison.md`：逐组结果、逐 seed 配对差值及配对差值的均值/样本标准差。

汇总会检查实际解析后的配置：除均值梯度开关与输出元数据外必须相同；两组优化步数、记录的依赖版本和 GPU 环境也必须相同。训练间代码或已记录输入变化、评估失败、缺失质量指标或未经验证的评估缓存都不会被当作成功的实验结果。输出目录必须是新目录，防止覆盖或混入旧实验。参考文件哈希验证两组使用同一份日志；参考模型和 split 是否匹配仍需在运行前核实。

全部差值均为 **stop_mean − full_mean**：FQ/MU/流畅度通常越高越好，KS statistic/EM 通常越低越好。FQ 是 KS 检验 p-value，不是“遗忘百分比”；同时看 KS statistic、EM、MU 与流畅度，避免只凭一个饱和的 p-value 下结论。

`train_runtime` 和每优化步秒数来自 Trainer，训练阶段关闭了评估。`entrypoint_wall_seconds` 另包含模型/数据初始化和最终保存，多卡取各进程耗时最大值。GPU 显存记录是整个训练入口的 PyTorch allocator 峰值，包括模型与参考模型加载，按字节记录；多卡汇总取各卡峰值最大值，同时保留每卡明细。它不等同于 GPU 总占用或各卡占用之和，第三方 CUDA 分配也可能不在其中。CPU 运行没有 GPU 显存数值。默认报告比较相同预算后的结果，不直接测量达到某一 FQ/MU 门槛的时间。

`detach()` 只改变反向传播路径；全词表均值前向、完整 LM 输出层和常规矩阵反向仍然存在。不能把尾部输出梯度归零等同于整个模型的稀疏计算，也不能把这项消融预先认定为改进。

已有输出可以单独重建汇总：

```bash
python scripts/palu_mean_ablation/run_comparison.py \
  --summarize artifacts/palu_mean_ablation/tofu_forget05
```

## 自动测试和本地验证范围

```bash
python -m unittest discover -s tests/palu_mean_ablation -v
```

测试覆盖解析梯度、128256 维词表、全词表选择时的等价性、尾部梯度恰好为零的例外、冻结参考模型选择、跨度前缀/EOS/mask、无敏感位置、同初始权重的两组微型 Llama 训练、真实 Hydra 训练/评估配置、配对汇总与无效评估拒绝，以及多卡启动计划、分卡遥测、ZeRO-3 参考模型配置和权重清理。

`--skip-eval` 仅用于训练流程或性能检查，报告明确标注没有测量遗忘效果。2026-10-10 本地验证使用 CPU、微型 Llama、合成 tokenizer、仓库 forget05 标注和本地小型 retain 数据，各组通过真实训练入口执行 2 步。正式 7B/8B TOFU 遗忘效果、真实多 GPU 执行与 CUDA 性能尚需在 GPU 环境运行。

本次测试数量、梯度结果和真实入口验证记录见 [VALIDATION.md](VALIDATION.md)。
