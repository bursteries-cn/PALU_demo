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

# 2026-10-11 启动、四条损失曲线及表征诊断验证

实现集中参数的 Bash 启动、实时子进程日志、每 optimizer step 的四项损失记录、固定训练样本的 Retain-only 参考表征缓存与周期测量，以及跨 seed 的逐点 95% t 区间端帽误差棒。默认参考位置是最终 RMSNorm 后、LM head 前的 h[a-1]，方向为 KL(softmax(h_R/T) ∥ softmax(h_U/T))；未把它加入训练目标。

在上述同一 CPU 依赖环境，消融目录 33 项自动测试通过，其中新增 10 项验证答案首 token（含 -200 标签和 padding）、KL 方向/维度求和、pre-head 提取、随机数/模块模式恢复、缓存篡改拒绝、梯度累积记录、真实 Hydra/Bash 启动（含空格和中文路径）、t 区间和单 seed 缺失区间、四曲线 PNG/PDF 输出。

微型 Llama 开启 attention dropout，训练随机抽取 Retain 样本并累积两个 microbatch；有/无周期诊断的最终参数逐张量完全一致。诊断间隔为 2、训练 3 步，确认 step 0/2/3 测量，最终测量在 Trainer 汇总 runtime 之前进行，参考模型没有梯度。

通过真实驱动、参考预计算进程和训练入口，运行 seed 0/1/2 × full_mean/stop_mean，各 3 步；保存四条损失、逐样本 KL/余弦、固定样本诊断损失、曲线统计及 PNG/PDF。两组每个图点均有 n=3 的统计，正常结束后的六份最终模型权重全部清理。输出为仓库 `artifacts/palu_mean_ablation/local_validation_diagnostics_20261011_v2/`，由 gitignore 排除。此次参考使用同一初始微型 checkpoint，以检查 step 0 KL=0；没有训练真正的 Retain95 参考，没有 TOFU 质量评估，图只用于验证记录与可视化流程。

另运行真实双进程 CPU/Gloo 检查：3 条 Forget、2 条 Retain 探针，在 2 个进程间等长补齐；每次统计仍为 3/2、重复补齐样本不计入，只有主进程历史，两个进程最终参数完全相同。原始临时输出 `/private/tmp/palu-diagnostics-gloo-20261011_v2/`。可复核同一检查：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/palu_mean_ablation/check_distributed_diagnostics.py
```

本机无 CUDA，双 CPU 进程不验证 ZeRO-3 参数分片、服务器显存或真实 8B 多 GPU 性能；使用 README 的 2 步命令在服务器检查。增加诊断后的 runtime 包含额外前向和汇总开销，不能与诊断关闭的旧性能结果直接混作梯度方案效率差异。
