# PALU 单组诊断基线验证

## 2026-10-13：experiment 层级与同组 seed 汇总

按用户确认，同组 seed 共用时间戳，新增默认路径 `artifacts/palu/<experiment>/<北京时间戳>/seed_<seed>`。相同配置且 seed 不重复的独立启动会加入最新匹配组；重复 seed、科学配置变化或 `NEW_GROUP=1` 另开时间戳。实验内容通过 bash 的 `EXPERIMENT` 设置。

- 全部 42 项 CPU 回归测试通过，日志 `/private/tmp/palu-experiment-layout-tests-20261013.log`。覆盖跨启动分组、GPU 编号来源记录、不同配置/源码分组、重复 seed 不覆盖、手动新组、并行 seed 注册和 manifest 更新不丢失。
- 随后补充“最新组已有重复 seed 时不得回填更旧组”的断言；6 项目录/分组测试通过，日志 `/private/tmp/palu-experiment-layout-targeted-20261013.log`。
- 通过实际启动、独立参考预计算和训练入口，分三次运行微型模型 seed 0/1/2，各 2 个 CPU 优化步。三次启动使用不同可见 GPU 编号（CPU 模式），均加入 `artifacts/palu/local_validation_seed_group_20261013/20261013_044336_319802/`，保存 3 份 seed 产物、3 条启动来源记录和同一份参考缓存，汇总状态 completed。
- 共 26 条逐步曲线统计均为 n=3，均值和点态 95% Student-t 置信区间完整；PNG/PDF 与实验名称/探针位置图注已检查。成功完成后的微型模型权重按默认设置清理。

本次没有重跑 8B 模型或新的 FQ/MU 实验。以下为此前探针及基线实现的历史验证。

## 2026-10-13：Forget 探针改为首个敏感 token

Forget 提取 `h[first_sensitive-1]`，Retain 保持 `h[first_answer-1]`。参考缓存升级为 schema 2，旧缓存明确拒绝；历史曲线仍按旧位置标注，新旧位置不能合并聚合。

- 全部 36 项 CPU 回归测试通过，日志 `/private/tmp/palu-sensitive-probe-tests-20261013.log`。覆盖普通前缀跳过、多个敏感片段只取最早 token、padding/EOS、缺失敏感 token 跳过及数量不足报错、旧缓存和错误索引拒绝、当前/参考位置一致、敏感 token 及其后文不影响 `h[s-1]`。
- 含随机 Retain 配对、dropout 和梯度累计的微型训练中，开启/关闭诊断后的最终参数逐位一致。
- 双 CPU rank / Gloo 验证通过，日志 `/private/tmp/palu-sensitive-probe-gloo-20261013.log`。首次沙盒内运行被 OpenMP 共享内存限制中止，允许本机共享内存/通信后通过；没有改变模型实现来绕过测试。
- 使用实际 `run_experiment.py`、独立 `prepare_diagnostics.py` 与训练入口，在 `artifacts/palu/local_validation_sensitive_probe_20261013/` 完成 seed 0 的 2 步微型 CPU 训练。缓存记录 Forget=`first_sensitive`、Retain=`first_answer`；step 0/1/2 的 KL/余弦与四条损失记录完整，PNG/PDF 输出及新图注已检查。参考使用同一微型初始模型，step 0 两组 KL 均为 0。

该验证没有重跑 8B 模型，没有新的 FQ/MU 结果；CUDA/ZeRO-3 实际运行仍需服务器验证。以下为此前基线版本的历史验证。

日期：2026-10-12。代码从原实验分支 `cca8424` 拆出，移除梯度切断与双组调度，保留官方均值梯度、训练记录和固定样本诊断。

## 本地回归测试

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests/palu -v
```

32 项测试通过，包括：

- 真实 PALU 损失的完整均值梯度公式、128256 维词表、冻结参考 top-K、敏感片段前缀/EOS/mask、无敏感位置及微型模型参数更新。
- 实际 Hydra 配置、每 seed 单次启动、跨 seed 配置/步数/运行环境校验、FQ 参考与评估模型来源校验。
- 多卡命令与有效 batch、单卡独立评估、逐 rank 遥测、ZeRO-3 参考配置、成功清理权重以及失败/符号链接保护。
- 首答案预测位置、最终归一化后的表征捕获、KL 方向、参考缓存校验、RNG/模式恢复。含 dropout、随机 Retain 配对和梯度累计时，开启/关闭诊断的最终参数逐位相同。
- 四条曲线的 PNG/PDF、95% Student-t 区间、单 seed 无虚构区间、缺失步不插值、中文/空格路径以及 bash 启动参数。

测试日志：本机 `/private/tmp/palu-baseline-tests-20261012.log`。

## 真实入口 CPU 冒烟运行

通过新的 `run_experiment.py`、独立参考预计算进程和实际 `src/train.py`，运行 seed 0/1/2，各 3 个优化步，梯度累计 2。每个 seed 只生成一个训练目录。开启四条损失记录和 Retain 表征诊断，探针为 4 个 Forget 与 2 个 Retain 样本，记录 step 0/2/3。

结果目录：仓库的 `artifacts/palu/local_validation_20261012/`，由 gitignore 排除。

- `summary.json` 状态为 completed，包含 3 个 seed 的绝对值汇总。
- 三个 seed 的最终 `model.safetensors` 均按默认设置删除，结果、配置及诊断历史保留。
- 生成单面板四条训练损失图，以及上下两个面板的固定损失/表征 KL 图；PNG/PDF 已生成并检查。30 条曲线统计记录均为 n=3。
- 此次参考采用同一初始微型 checkpoint，以验证 step 0 KL=0；没有训练真实 Retain95，也没有执行 TOFU 质量评估。这是工程流程验证，不能用于比较遗忘效果或 GPU 性能。

## 双进程诊断验证

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/palu/check_distributed_diagnostics.py
```

本机使用真实双 CPU rank / Gloo，训练 2 步并在 step 0/1/2 诊断。通过参数一致性、主进程单份历史、3 个 Forget / 2 个 Retain 的不均匀填充排除检查。输出 `/private/tmp/palu-baseline-gloo-20261012`。

本轮环境为 macOS CPU，PyTorch 2.4.1、Transformers 4.45.1；未执行 CUDA/DeepSpeed ZeRO-3 的实际训练。服务器多 GPU 数值表现与资源开销仍以服务器运行结果为准。

原双组消融的验证记录与用户服务器实验由原分支和独立 memory 笔记保留，不作为本次代码重构的新增实验结果。
