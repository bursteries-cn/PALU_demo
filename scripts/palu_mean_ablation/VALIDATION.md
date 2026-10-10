# 2026-10-10 本地验证

结论：全词表均值的梯度差异已通过真实 PALU 损失验证，两组真实训练入口也均可运行。尚未测量正式 7B/8B 模型的遗忘质量或 CUDA 性能。

代码基于 `16cb23de5c168e2b1efcb30665c5bc78ebc997c8`；实验分支为 `experiment/palu-mean-gradient-ablation`。

## 自动测试

在 Python 3.11.15、torch 2.4.1、transformers 4.45.1、accelerate 0.34.2、datasets 3.0.1 的 CPU 环境验证：

| 测试组 | 通过数量 |
|---|---:|
| 新增均值梯度、微型训练与对比协议测试 | 15 |
| 原有 representation_batching 测试 | 62 |
| 原有超参数分析测试 | 11 |
| 原有 NPO bash launcher 测试 | 7 |
| 合计 | 95 |

按测试目录分别执行。仓库原有 tests/representation_batching 与 src/representation_batching 同名，根目录统一 unittest discovery 会出现导入冲突。本机系统 Python/Git 也受 Xcode license 限制，验证使用临时项目依赖环境及已有 fallback Git；未修改这些系统设置。

## 梯度探针

真实 `PALU.compute_loss`、logits `[5, 3, 0, -2, -1]`、K=2、仅敏感遗忘项：

| 实验组 | loss | 梯度 |
|---|---:|---|
| full_mean | 10 | `[2.8, 0.8, -1.2, -1.2, -1.2]` |
| stop_mean | 10 | `[4, 2, 0, 0, 0]` |

测试另验证 V=128256、K=5000 的解析公式及 123256 个词表外维度的梯度差异。

## 真实训练入口

两组共用本地微型 Llama checkpoint、合成 tokenizer、仓库 forget05 标注、本地小型 retain 数据和 seed=7。K=10、first_n=2、lr=0.001、batch=2、accumulation=1；各通过 telemetry wrapper 调用真实 `src/train.py`，执行 2 个优化步骤并保存最终模型。关闭评估，以验证流程。

| 检查 | 结果 |
|---|---|
| 两组首步日志 loss | 均为 1.5564 |
| full_mean 首步参数梯度范数 | 0.6777478456497192 |
| stop_mean 首步参数梯度范数 | 0.6905940175056458 |
| 两组最终参数不同的张量数 | 12 |
| 最终参数最大绝对差值 | 0.0029981923289597034 |
| 冻结参考模型不变 | 微型 Trainer 自动测试通过 |
| 配置、seed、优化步数和环境一致性 | 配对汇总通过 |
| CUDA 可用 | 否 |

本地原始输出位于仓库根目录的 `artifacts/palu_mean_ablation/local_validation_20261010/`，由 gitignore 排除，包含模型、配置、日志、梯度探针与对比表。短步 CPU 耗时不能用于估计 GPU 加速幅度；该运行没有 FQ、MU、EM 或流畅度结果。

正式实验的运行方式与指标解释见 [README](README.md)。
