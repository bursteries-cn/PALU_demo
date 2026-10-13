# PALU + EfficientSoup 首轮实验

基于 `experiment/palu-diagnostics-baseline` 的 `f0a7196`，分支 `feature/palu-efficient-soup`。
训练损失、敏感 token 选择与原 PALU 基线保持一致，新增独立的训练后权重插值模块。

## 方法与适用范围

来源：[UnlearningSoup, arXiv:2609.37076v1 §4、Figure 4、Appendix D.3/E.4](https://arxiv.org/html/2609.37076v1)。
论文支持的流程是：同一个 Full 初始化的两个不同 seed 遗忘模型；先按 Forget/Retain ES 的组合分数选较好的候选；搜索 Full—较好候选这条边；再搜索第一阶段最优点—另一个候选这条边。

设 `a` 为第一条边中较好候选的比例、`b` 为第二条边中另一个候选的比例：

```text
theta_edge = (1-a) theta_full + a theta_better
theta_soup = (1-b) theta_edge + b theta_other
DS_ES = sqrt(ES_forget^2 + (1 - ES_retain)^2), lower is better
```

代码始终保存相对于 `[Full, candidate_1, candidate_2]` 的三个系数，非负且和为 1。
DS 同时保存原始值与乘 100 的值，不涉及 PerformanceSoup 的性能权重。

本版每轮评价区间中点，用中点替换当前较差的端点，同时独立保留已评价的最优点。
端点同分时替换右端点；候选同分时选择第一个候选；新点同分时保留已有点。
默认每条边 3 个中点，最多 `3 + 2*3 = 9` 次代理评估，含 Full 与两个候选；重复系数仅在本次运行内复用。
论文 Figure 4 给出了中点路径，但没有完整定义所有区间更新、平局及预算细节；附录 E.4 写的是 TOFU 8 次代理评估。
因此本版是**按论文思路实现的 PALU 适配实验**，不是官方代码的严格复现，也不沿用其速度倍数。
保留最优点只保证所用代理集上不差于已评价候选，不能保证真实遗忘效果、最终 FQ/MU 或全局最优。

## 一键训练和评估

在已有 PALU CUDA 环境中，从仓库根目录运行。首先检查脚本顶部的模型、数据、Retain 日志及分类器路径：

```bash
bash scripts/palu/run_efficient_soup.sh --dry-run
bash scripts/palu/run_efficient_soup.sh
```

默认参数：Llama-3.1-8B-Instruct / Forget05，seed `0 1`，PALU 使用现有 `run.sh` 的训练超参数；
训练用 `GPU_IDS=0,1,2,3`，Soup 与最终评估仅用其中第一张卡；强制保留两个最终 checkpoint。
本实验默认 `DIAGNOSTICS=0`，避免 Retain95 隐藏表征诊断开销，`LOSS_HISTORY` 沿用现有默认值。
Retain95 **评估日志**用于 FQ；Retain95 **模型权重**不是 Soup 原始锚点。

```bash
GPU_IDS=2,3 SEEDS="0 1" LR=2e-5 ALPHA=0.2 \
SOUP_DEPTH=3 PROXY_QUESTIONS=64 \
bash scripts/palu/run_efficient_soup.sh
```

其他可覆盖项：`PYTHON_BIN`、`MODEL`、`MODEL_PATH`、`MODEL_REVISION`、`DATASET_PATH`、`RETAIN_LOGS`、
`CLASSIFIER_MODEL`、`FORGET_SPLIT`、`RETAIN_SPLIT`、`HOLDOUT_SPLIT`、`PROXY_SEED`、`SOUP_RUN_DIR`，
以及 `run.sh` 已有的训练参数。
`FULL_EVAL=0` 仅运行代理搜索与留出 ES，适合冒烟检查；不能据此给出最终遗忘质量结论。
`SEEDS` 必须恰好包含两个不同整数。已存在的输出目录会报错，禁止覆盖；dry-run 也使用独立目录。

原模型可以是本地目录或 Hub ID；两个候选必须是完整的本地 safetensors checkpoint。
可使用 `MODEL_REVISION` 固定 Hub revision，降低同名 Hub 模型变化的风险。
分片权重、相同架构的 tied weights 都支持；不同 tokenizer/config、LoRA adapter、量化权重会拒绝。
若候选带 `resolved_config.json`，校验 PALU、不同 seed、其余科学设置、初始化路径、revision 和 Forget split。
没有该文件的手动 checkpoint 会明确记录来源未验证，需要使用者确保同一初始化和训练协议。

## 已有两个候选时直接搜索

必须在原训练时使用 `KEEP_MODEL=1`；评估 JSON 不能代替模型权重。
下面路径是示例，替换为实际保存目录与初始化路径：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/palu/efficient_soup.py \
  --original /path/to/full \
  --candidates /path/to/seed_0 /path/to/seed_1 \
  --dataset /path/to/TOFU \
  --model Llama-3.1-8B-Instruct \
  --output artifacts/palu_efficient_soup/manual_trial \
  --proxy-questions 64 --depth 3 \
  --full-eval --retain-logs /path/to/retain95/TOFU_EVAL.json \
  --classifier-model /path/to/gibberish-detector
```

CLI 另支持 `--forget-config`、`--retain-config`、`--split-seed`、`--max-length`、`--batch-size`、
`--device`、`--dtype`。现阶段要求每条数据的 `question` / `answer` 为字符串。

## 评估口径与对照

固定抽取 `forget05_perturbed` 与 `retain_perturbed` 各 64 个不同问题作为搜索代理集，余下问题只在选择完成后评价。
相同问题的重复记录放在同一组；这不是作者级留出，也不能保证训练数据未见。
所有模型使用完全相同的 tokenizer、模型配置中的 PALU 模板、问题、原始完整答案和位置掩码。
保存具体索引、逐条文本/分词哈希与模板，不随候选重新抽样。
不会只对 PALU 的敏感 token 计算 ES。

ES 使用 teacher forcing 下答案 token 的 argmax，统计最后一个预测错误位置之后的完整正确后缀长度占比。
计入已有预处理函数生成的 EOS/模板终止 token；空答案标签报错，超长序列报错，要求显式增大长度。
本地原有 `src/evals/metrics/memorization.py` 在最后一个 token 预测错误时仍返回 `1/n`，没有正确表示空后缀。
新模块在该情况下返回 0；未改动旧评估器，旧 ES 数值不能直接拼接进本版搜索报告。
批处理按真实长度生成 attention mask，避免 PAD 等于 EOS 时把 EOS 错屏蔽。

固定报告五组：Full、PALU seed 0、PALU seed 1、第一阶段插值、最终 Soup。
选择完成后才计算这五组的留出 ES，不能根据留出分数重新挑权重。
默认另外运行相同的完整 TOFU：FQ、KS statistic、MU、EM、流畅度及原始评估 JSON 中的分项。
标准 TOFU 全集含部分搜索问题，只作与既有基线一致的描述性评估，不冒充独立测试。
FQ 是 KS 检验 p 值；ES/DS 的改善不能证明知识删除。

首轮重点检查：

1. Soup 相对较好的单 seed 是否改善留出 ES 与完整 FQ/MU 权衡。
2. 相对第一阶段插值是否进一步改善，以区分两个候选融合与单纯减弱遗忘强度。
3. Forget 的信息提取是否随 Full 比例上升而恢复；保留提升可能同时损害遗忘。
4. 若有收益，再做多个独立 seed pair 及按作者留出的验证。两个 seed 合成的一锅不是两次独立实验，不能据此给 Soup 画 seed 置信区间。

## 输出与成本

```text
artifacts/palu_efficient_soup/forget05/<time_pid>/
  candidates/manifest.json
  candidates/seed_0/                 # 原始训练结果及完整权重
  candidates/seed_1/
  soup_search/manifest.json          # 协议、输入/代码 SHA256、环境、耗时、状态
  soup_search/samples.json           # 搜索/留出索引及样本哈希
  soup_search/search.jsonl           # 每次实际搜索的权重、逐样本 ES、耗时
  soup_search/selection.json         # 在查看留出集之前固定的选择
  soup_search/heldout.json
  soup_search/anchor_edge/           # 第一阶段模型
  soup_search/soup/                  # 最终模型，可直接给 src/eval.py
  soup_search/full_eval_commands.json
  soup_search/full_eval_summary.json # 含 FQ / KS D / MU / EM / 流畅度
  soup_search/full_eval/<model>/     # 完整评估及其来源记录
```

GPU 只加载一个模型；CPU 从 safetensors 映射的原始权重逐张量以 FP32 累加，再转回目标 dtype。
每个搜索点均重新从原始三份权重构造，避免低精度递归插值的累计漂移。
CPU 仍需要模型加载、最大张量的工作空间和操作系统文件缓存；合并涉及磁盘读取与 CPU→GPU 传输。
8B bf16 每份权重约 16 GB：原始 Full、两个候选和两个导出模型合计约 80 GB，另留日志/缓存空间。
本版不会自动删除模型。

记录初始化与权重哈希、合并、代理评估、保存、完整评估耗时；训练耗时保留在候选运行记录中。
`soup_total_wall_seconds` 不含候选训练。首轮五组完整评估是额外的诊断成本，不能将其与论文的一次最终评估直接比较。
只有在相同硬件、目标质量与评估预算下汇总训练/加载/合并/评估的全部成本后，才能判断节省多少时间。

完整 TOFU 阶段会在启动日志显示 `[full_eval 1/5]` 等阶段提示，并同步输出各指标的实时进度；
同样的详细输出保存在 `full_eval_<model>.log`。旧版本 `8b1af12` 的完整评估仅写独立日志，
启动日志长时间停在 DeepSpeed 初始化提示不代表任务停止。已启动的旧进程应查看独立评估日志，
无需为日志显示而重启实验。

## 验证范围

```bash
python -m unittest discover -s tests/palu -p test_efficient_soup.py -v
python -m unittest discover -s tests/palu -v
```

测试覆盖 ES 边界及逐步贪心生成对照、PAD=EOS、两阶段搜索预算与最优点保留、分片/tied weights、
重新从源权重构造、缺失权重/非浮点 buffer 失败、离线微型 Llama 的完整搜索与保存重载、问题去重分组及禁止覆盖。
本地 CPU 验证不能代替 8B CUDA 实跑，也不说明 PALU + EfficientSoup 已取得性能收益。

2026-10-13 本地验证：51 项 PALU/Soup 测试通过（含新增 9 项），一键入口 dry-run、Bash 语法检查通过。
微型 Llama 的离线搜索、留出评价、分片/tied weights 与导出模型重载已实际执行。
CUDA/DeepSpeed 的两候选训练和 8B 完整 TOFU 尚未在本次工作中运行。
