# 2026-10-10 本地验证

结论：全词表均值的梯度差异已通过真实 PALU 损失验证，两组真实训练入口也均可运行。尚未测量正式 7B/8B 模型的遗忘质量或 CUDA 性能。

代码基于 `16cb23de5c168e2b1efcb30665c5bc78ebc997c8`；实验分支为 `experiment/palu-mean-gradient-ablation`。

## 自动测试

在 Python 3.11.15、torch 2.4.1、transformers 4.45.1、accelerate 0.34.2、datasets 3.0.1 的 CPU 环境验证：

| 测试组 | 通过数量 |
|---|---:|
| 均值梯度、微型训练、对比协议、多卡配置与清理测试 | 23 |
| 原有 representation_batching 测试 | 62 |
| 原有超参数分析测试 | 11 |
| 原有 NPO bash launcher 测试 | 7 |
| 合计 | 103 |

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

## 默认清理与多卡支持

多卡配置测试验证：四卡 Accelerate/ZeRO-3 启动、第一张卡独立评估、1 × 8 × 4 = 32 的有效 batch、两组真实 Hydra 配置仅在均值梯度开关与输出元数据上不同。模拟各卡遥测验证分进程文件和最大值汇总；模拟 CUDA 接口验证加载前选择 LOCAL_RANK 对应设备。参考模型初始化测试验证 ZeRO-3 的三个 bucket 设置写入嵌套 `zero_optimization` 并解析为整数，且不修改训练模型的共享配置。

权重清理测试验证：成功结果删除模型分片及索引、保留日志/配置/tokenizer/训练状态；`--keep-model` 保留权重；未完成或失败结果保留权重；符号链接不会被清理；评估其他 checkpoint 的结果会被拒绝。

使用同一微型 checkpoint 另跑一次完整驱动程序，显式 `--gpu 0 --skip-eval`，两组各完成 2 个优化步骤。两组 `model.safetensors` 均按默认设置删除，`result.json` 记录 `keep_model_weights=false` 和删除文件名；训练状态、解析配置及比较结果保留。删除后通过 `--summarize` 重新生成汇总。原始输出为 `artifacts/palu_mean_ablation/local_validation_nosave_20261010/`。该 CPU 运行没有测量遗忘质量。

以上多卡检查属于配置及模拟接口测试。本机没有 CUDA，未实际启动 DeepSpeed 多 GPU 训练，不能据此确认服务器的显存、通信或实际训练速度。

正式实验的运行方式与指标解释见 [README](README.md)。
