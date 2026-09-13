# NPO 基线运行入口

当前使用独立的 OpenUnlearning 固定提交，测试分支为 `test/openunlearning-npo-pinned`。

使用 **`train_tofu_npo_official.sh`**，完整步骤见 [6000D 官方代码测试说明](NPO_OFFICIAL_6000D.md)。默认四卡、SDPA；训练与评估都从 `--upstream-dir` 的干净官方 checkout 运行。硬件适配模式记录 torch、bitsandbytes、deepspeed 的版本差异，其余官方固定依赖仍须匹配。不要把官方 torch=2.4.1 直接装回 6000D 环境。

| 入口 | 用途 |
| --- | --- |
| `train_tofu_npo_official.sh` / `run_official_npo.py` | 当前官方固定提交测试，使用独立目录和环境。 |
| `train_tofu_npo_llama3.sh` | 历史本地 NPO 路径，使用 PALU_demo 自己的训练代码。 |
| `../../representation_batching/run_seed.sh` | R/S/D/P 分组实验，使用 RepresentationNPO 与离线 manifest。 |

## 历史本地 NPO 入口

该历史入口使用本仓库的 NPO 和现有 attention 配置，不安装或检查 FlashAttention-2，也不要求额外 clone 官方仓库。

激活服务器现有 PALU/NPO 环境后，在 PALU_demo 根目录执行：

```bash
# 只打印命令，不启动训练、不创建结果目录
bash scripts/unlearn/tofu/train_tofu_npo_llama3.sh --dry-run

# 使用四张卡直接训练，并自动做最终评估
bash scripts/unlearn/tofu/train_tofu_npo_llama3.sh
```

默认参数为 GPU `0,1,2,3`，**4 卡 × 每卡 batch 4 × 梯度累积 2 = 32**，seed=0，lr=1e-5，10 epochs，1 epoch warmup、linear scheduler、paged_adamw_32bit、weight decay=0.01、beta=0.1、alpha=gamma=1。`max_grad_norm=0` 显式关闭本次训练的裁剪；可用 `--max-grad-norm 1` 开启。该参数通过现有 DeepSpeed 的 `gradient_clipping: auto` 生效，不改其他实验的配置。

默认沿用原服务器路径：

- 模型：`open-unlearning/tofu_Llama-3.1-8B-Instruct_full`。
- 数据：`/mnt/sda/cr/LLM_unlearning/datset/TOFU`。
- 参考日志：`saves/eval/tofu_Llama-3.1-8B-Instruct_retain95/TOFU_EVAL.json`。
- 结果：`saves/unlearn/tofu/forget05/Llama-3.1-8B-Instruct/npo_baseline/<运行名>/`。

这些文件在服务器其他位置时，可直接覆盖路径：

```bash
bash scripts/unlearn/tofu/train_tofu_npo_llama3.sh \
  --gpu 0,1,2,3 --seed 1 --epochs 10 \
  --dataset /path/to/TOFU \
  --retain-logs /path/to/retain95/TOFU_EVAL.json \
  --output-dir /path/to/results/npo-four-gpu/seed-1
```

`--output-dir` 指定一个全新的精确目录；`--output-root` 指定多次运行/参数扫描的父目录，脚本自动创建子目录。原有 `--lr`、`--beta`、`--alpha`、`--gamma` 的多值扫描仍可使用，例如 `--lr "1e-5 2e-5"`；扫描时使用 `--output-root`。

执行顺序是：四卡训练 → 临时保存最终模型 → 所选第一张 GPU 独立评估 → 校验四项结果和评估来源 → 默认删除权重。`--keep-model` 保留权重，`--no-save` 表示评估后清理，运行过程中仍会临时保存。训练或评估失败会退出并保留已有权重、日志；不会因 `tee` 写日志而吞掉训练错误。

每个结果目录保存 `launch_commands.sh`、`resolved_config.json/yaml`、`train.log`、`eval.log`、`trainer_state.json`、`evals/TOFU_EVAL.json` 和 `evals/TOFU_SUMMARY.json`。全部通过后写 `npo_run_complete.json`。普通 NPO 的 Forget/Retain loss 可在训练日志和 Trainer 的 log_history 中查看。

该入口使用当前本地环境（requirements 中 Transformers=4.45.1），不属于独立官方 checkout 的逐项数值复现。四卡分片、梯度累积和 attention 后端与官方双卡路径存在差异；尾部累积窗口、样本覆盖和实际更新次数仍属于后续 GPU 审计范围。这里没有修改 Trainer 的尾部处理或采样算法。
