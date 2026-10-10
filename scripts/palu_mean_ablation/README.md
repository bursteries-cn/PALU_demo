# PALU 全词表均值梯度消融

从远端最新分支 `origin/feature/npo-representation-batching` 的 `16cb23de5c168e2b1efcb30665c5bc78ebc997c8` 创建，实验分支为 `experiment/palu-mean-gradient-ablation`。

这个实验只改变当前模型全词表均值是否参与反向传播，比较两条优化路径：

| 实验组 | Hydra 配置 | 含义 |
|---|---|---|
| `full_mean` | `trainer.method_args.detach_mean_target=false` | 默认，保留官方行为 |
| `stop_mean` | `trainer.method_args.detach_mean_target=true` | 对同一个全词表均值执行 `detach()` |

两组的均值仍随当前模型变化。在相同 logits 上，前向损失完全相同；参数更新后，两组的模型、均值和损失可以逐渐分化。`stop_mean` 不是固定参考模型均值，也不是 top-K 均值。

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

默认设置沿用 Llama-3.1 demo：forget05/holdout05/retain95、K=5000、每个敏感跨度 first_n=3、lr=2e-5、alpha=0.2、gamma=1、batch=8、累积=4、10 epochs、SDPA、梯度检查点。两组都使用 NLL 保留项与原有普通位置约束。默认 seed 为 42、43、44。

先生成计划，检查模型和参考路径：

```bash
python scripts/palu_mean_ablation/run_comparison.py \
  --gpu 0 --seeds 42 43 44 \
  --model-path /absolute/path/to/tofu_full_model \
  --retain-logs /absolute/path/to/retain95/TOFU_EVAL.json \
  --output artifacts/palu_mean_ablation/plan \
  --dry-run
```

正式运行用新的输出目录：

```bash
python scripts/palu_mean_ablation/run_comparison.py \
  --gpu 0 --seeds 42 43 44 \
  --model-path /absolute/path/to/tofu_full_model \
  --retain-logs /absolute/path/to/retain95/TOFU_EVAL.json \
  --output artifacts/palu_mean_ablation/tofu_forget05
```

模型路径可以是本地完整 checkpoint 或 Hub id；使用 Hub id 时，可用 `--model-revision <commit>` 固定模型和 tokenizer 的版本。Llama-2 chat 可通过 `--model Llama-2-7b-chat-hf` 选择，参考日志必须与模型和 split 匹配。

离线评估支持 `--eval-dataset /absolute/path/to/TOFU` 和 `--classifier-model /absolute/path/to/fluency_classifier`。后者只指定已有评估器的流畅度分类器。共享训练参数可用重复的 `--override` 传入，例如：

```bash
--override trainer.args.optim=adamw_torch
```

需要比较固定优化步数时，用 `--max-steps 100`。单 GPU 显存不足时，同时降低两组的 `--batch-size` 并增加 `--grad-accum`，保留相同有效 batch。脚本按单进程运行，避免仓库在多进程训练时跳过自定义评估。每个实验组都会保留最终模型；默认三组 seed 共保存六个模型，需预留磁盘空间。

## 输出与实验有效性

每个实验组使用独立 Python 进程，从相同 checkpoint 开始；训练 seed 与数据采样 seed 一致。不同 seed 交替组别顺序，降低固定运行顺序的影响。训练时关闭评估，保存最终模型后另起进程运行 TOFU 评估。

脚本保存：

- `manifest.json`：精确命令、共同设置、代码与可见本地输入文件的 SHA-256。
- `gradient_probe.json`：真实遗忘损失的梯度验证。
- `seed_<seed>/<variant>/resolved_config.json`、`trainer_state.json`、训练日志和最终模型。
- `telemetry.json`：实际依赖版本、GPU 名称、CUDA 分配/保留显存峰值和训练入口耗时。
- `evals/TOFU_EVAL.json`、`TOFU_SUMMARY.json`、评估来源记录。
- `comparison.json`、`comparison.csv`、`comparison.md`：逐组结果、逐 seed 配对差值及配对差值的均值/样本标准差。

汇总会检查实际解析后的配置：除均值梯度开关与输出元数据外必须相同；两组优化步数、记录的依赖版本和 GPU 环境也必须相同。训练间代码或已记录输入变化、评估失败、缺失质量指标或未经验证的评估缓存都不会被当作成功的实验结果。输出目录必须是新目录，防止覆盖或混入旧实验。参考文件哈希验证两组使用同一份日志；参考模型和 split 是否匹配仍需在运行前核实。

全部差值均为 **stop_mean − full_mean**：FQ/MU/流畅度通常越高越好，KS statistic/EM 通常越低越好。FQ 是 KS 检验 p-value，不是“遗忘百分比”；同时看 KS statistic、EM、MU 与流畅度，避免只凭一个饱和的 p-value 下结论。

`train_runtime` 和每优化步秒数来自 Trainer，训练阶段关闭了评估。`entrypoint_wall_seconds` 另包含模型/数据初始化和最终保存。GPU 显存记录是整个训练入口的 PyTorch allocator 峰值，包括模型与参考模型加载，按字节记录；不等同于 GPU 总占用，第三方 CUDA 分配也可能不在其中。CPU 运行没有 GPU 显存数值。默认报告比较相同预算后的结果，不直接测量达到某一 FQ/MU 门槛的时间。

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

测试覆盖解析梯度、128256 维词表、全词表选择时的等价性、尾部梯度恰好为零的例外、冻结参考模型选择、跨度前缀/EOS/mask、无敏感位置、同初始权重的两组微型 Llama 训练、真实 Hydra 训练/评估配置、配对汇总与无效评估拒绝。

`--skip-eval` 仅用于训练流程或性能检查，报告明确标注没有测量遗忘效果。2026-10-10 本地验证使用 CPU、微型 Llama、合成 tokenizer、仓库 forget05 标注和本地小型 retain 数据，各组通过真实训练入口执行 2 步并保存模型。正式 7B/8B TOFU 遗忘效果与 CUDA 性能尚需在 GPU 环境运行。

本次测试数量、梯度结果和真实入口验证记录见 [VALIDATION.md](VALIDATION.md)。
