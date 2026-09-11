# NPO 官方基线入口

当前分支的三个入口：

| 入口 | 用途 |
| --- | --- |
| `train_tofu_npo_llama3.sh` | 早期移植的普通 NPO。默认单卡、lr=2e-5、batch=8、累积=4、10 epochs；不是已经对齐官方的基线。 |
| `../../representation_batching/run_seed.sh` | 当前 R/S/D/P 分组实验，使用 RepresentationNPO 和离线 manifest。 |
| `run_official_npo.py` | 独立调用下述固定版本的官方仓库，只运行 Llama-3.1-8B、forget05/retain95 的一个 NPO seed。 |

## 固定的来源与设置

- 上游：<https://github.com/locuslab/open-unlearning>
- 提交：`4ad738aaf60f6a4385f6e2506d01da99e76c31f3`
- 来源脚本：`scripts/tofu_unlearn.sh`，实际每卡 batch=4、累积=4、双卡，总 batch=32。
- 普通 `NPO` + 官方 DataLoader，无 RepresentationNPO、表征缓存或 manifest。
- lr=1e-5、warmup_epochs=1、linear scheduler、10 epochs、paged_adamw_32bit。
- beta=0.1，alpha=gamma=1，bf16，FlashAttention-2，ZeRO-3。
- 使用上游的 DeepSpeed JSON，保留其未配置 gradient_clipping 的行为。
- 模型 `open-unlearning/tofu_Llama-3.1-8B-Instruct_full`，数据 `locuslab/TOFU`。

脚本不安装依赖、不修改官方源码、不导入本地 PALU 的 trainer/config。相对于官方批量脚本，只选择一组模型/数据/算法，暴露 seed、GPU、端口、结果目录和 Retain 参考日志，并强制重新计算最终评估。双卡训练后，由所选第一张 GPU 独立评估。

## 在服务器准备独立目录和环境

不要在 PALU 的现有环境里直接升级 Transformers。以下路径是示例，按服务器实际位置修改：

```bash
git clone https://github.com/locuslab/open-unlearning.git /path/to/open-unlearning-npo
git -C /path/to/open-unlearning-npo checkout --detach 4ad738aaf60f6a4385f6e2506d01da99e76c31f3
```

依据该提交的 README/setup.sh 在独立环境安装依赖；运行入口会检查官方 requirements.txt 的固定版本，以及 FlashAttention-2 和两张可见 CUDA 卡。不自动创建/替换服务器环境。

`--retain-logs` 必须是同一模型的 Retain95 在匹配协议下产生的 `TOFU_EVAL.json`，其中包含 `forget_truth_ratio`。不是 `TOFU_SUMMARY.json`，也不是 NPO 自身的结果。脚本可以检查结构和记录 SHA256，但文件名不能证明它对应的模型、split 和数据版本，实际来源仍需核对。

## 第一步：dry-run 后运行官方基线

在 PALU_demo 根目录运行，替换三个 `/path/to/...` 占位路径：

```bash
python3 scripts/unlearn/tofu/run_official_npo.py \
  --upstream-dir /path/to/open-unlearning-npo \
  --python /path/to/npo-official-env/bin/python \
  --retain-logs /path/to/retain95/TOFU_EVAL.json \
  --output-dir /path/to/results/npo-official/seed-0 \
  --gpu 0,1 --seed 0 --dry-run
```

检查命令后去掉 `--dry-run` 才执行。dry-run 本身不检查服务器环境、不下载模型或数据，也不创建结果目录；实际启动会检查固定提交、源码干净状态、依赖、GPU 和参考日志。结果目录必须是新目录，且位于官方 checkout 外。

默认在训练、保存、独立评估及结果验证全部成功后删除本次生成的权重，保留 JSON、配置和日志。加 `--keep-model` 则保留权重；评估失败不会清理权重。官方 train.py 会保存最终模型，因此运行过程中仍需要权重的临时磁盘空间。

结果结构：

```text
seed-0/
  baseline_audit/
    run_contract.json       # 上游提交、命令、环境、参考日志 SHA256
    resolved_config.yaml   # 上游 Hydra 实际解析的训练配置
    train.log
    eval.log
    status.json            # 状态、计划更新完成情况、实际 global_step/epoch
  trainer_state.json
  evals/
    TOFU_EVAL.json          # 原始逐样本/派生评估数据
    TOFU_SUMMARY.json       # 官方启用的汇总指标
```

`completed` 只表示本次训练/评估流程及文件检查通过，不代表复现官方论文数值，也不代表数值稳定。脚本不把官方 Trainer 自身的计划更新数当作训练协议正确性的证明；仍需核对实际 epoch、样本覆盖以及 DeepSpeed 更新数。

## 第二步：首个优化步骤与最终结果核对

本入口保留未插入诊断 hook 的官方执行路径，尚不额外记录每个 microbatch 的 Forget/Retain loss 或样本 ID。以下检查属于后续审计，不能因为入口已经创建就标记完成：

1. 记录模型/数据的实际 Hub revision 或本地 snapshot；对齐 Full 与 Retain95 参考结果。
2. 对固定 seed 做首个优化步骤的独立诊断重放：样本 ID、Forget/Retain loss、NPO weight、实际送入 backward 的 loss、梯度累积缩放、裁剪阈值与更新范数。诊断不得额外抽样或额外执行训练 forward。
3. 核对 Trainer global_step、DeepSpeed 实际更新次数及每个 epoch 的样本覆盖。尤其检查 200 条 Forget 与有效 batch=32 的尾部累积窗口。
4. 核对当前上游 Transformers 4.51.3 与自定义 NPO loss 的 `num_items_in_batch`/`model_accepts_loss_kwargs` 接口。这是待验证项，不能先验宣称当前上游没有兼容性问题。
5. 用相同 evaluator 比较 Full、Retain95 与 NPO 的结果。

## 第三步：再接回随机组 R

完成官方路径运行及上述检查后，再在相同模型、环境、优化器、microbatch、学习率调度和评估配置下替换采样器。当前 R 的 manifest 强制 Forget 样本数整除有效 batch，因此不能把 20 简单改成 32，也不能为了适配而静默丢弃样本。不要在同一次对照里同时改变采样器和训练超参数。

当前交付状态：启动入口与离线检查；尚未执行服务器 GPU 训练、首步梯度审计或数值复现。
