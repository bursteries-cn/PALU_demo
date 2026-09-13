# 在 6000D 上测试固定提交的 OpenUnlearning NPO

当前测试分支：`test/openunlearning-npo-pinned`。目标是运行 **OpenUnlearning 项目维护的 NPO 实现**。

- 代码来源：[locuslab/open-unlearning](https://github.com/locuslab/open-unlearning/tree/4ad738aaf60f6a4385f6e2506d01da99e76c31f3)。
- 固定提交：`4ad738aaf60f6a4385f6e2506d01da99e76c31f3`。
- NPO 原论文作者的实验代码另在 [licong-lin/negative-preference-optimization](https://github.com/licong-lin/negative-preference-optimization)。两者是不同的复现目标。
- `train_tofu_npo_official.sh` 调用独立官方目录里的 `src/train.py`、`src/eval.py` 和 configs。PALU_demo 只提供启动与结果记录，不修改官方源码。

## 硬件适配的范围

6000D 属于 Blackwell 系列，不能直接照搬该提交中的 `torch==2.4.1`。PyTorch 从 2.7 的 CUDA 12.8 构建开始提供 Blackwell 支持；这不意味着只升级 torch 就能保证 DeepSpeed 和 bitsandbytes 也兼容。[PyTorch 官方说明](https://pytorch.org/blog/pytorch-2-7/)

默认 `--environment hardware-compatible`：允许已安装的 **torch、bitsandbytes、deepspeed** 偏离上游固定版本，逐项记录实际版本和差异；缺包仍会报错。其余 requirements 固定版本仍须匹配，包括 Transformers=4.51.3、Accelerate=0.34.2、huggingface-hub=0.36.0。该提交的 evaluator 无条件导入 lm-eval，因此也检查 setup.py 中的 `lm-eval==0.4.11`。

优先在独立环境里保留旧环境中已经能训练的 GPU 依赖，再对齐其余依赖。这个策略减少未经验证的硬件栈变动，不把旧环境的成功当作新组合已经兼容。实际版本要根据服务器输出确定，并经过短训练核验。

`--environment official` 则检查全部官方版本；6000D 不应使用这一模式去强行降级 torch。脚本不自动安装、升级或降级任何依赖。

## 服务器准备

以下路径均需替换为服务器真实位置。本地已有的官方 checkout 不会随着 PALU_demo 的 Git 分支自动传到服务器。

```bash
NPO_UPSTREAM=/path/to/open-unlearning-npo
git clone https://github.com/locuslab/open-unlearning.git "$NPO_UPSTREAM"
git -C "$NPO_UPSTREAM" checkout --detach 4ad738aaf60f6a4385f6e2506d01da99e76c31f3
git -C "$NPO_UPSTREAM" rev-parse HEAD
```

如果服务器已有该目录，核对 HEAD 和工作区后使用它。启动器拒绝错误的提交或被改动的训练、评估、配置及依赖源文件。

先在当前能训练的旧环境记录硬件和依赖：

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python3 -c "import sys, torch; from importlib.metadata import version; print('Python:', sys.version); print('GPU:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print('Torch CUDA:', torch.version.cuda); print({p: version(p) for p in ('torch', 'transformers', 'accelerate', 'deepspeed', 'bitsandbytes')})"
```

### 使用 uv 从旧环境重建

先确认旧环境 Python 满足该提交的 `>=3.11` 要求。下面的流程以满足要求、已经能在 6000D 上训练的旧环境为基础；如果不满足，另建 Python 3.11 环境并重新选择兼容的依赖组合。

```bash
NPO_OLD_ENV=/path/to/old-env
NPO_ENV=/path/to/npo-official-6000d-env
NPO_PREP=/path/to/npo-environment-records
mkdir -p "$NPO_PREP"

uv pip freeze --python "$NPO_OLD_ENV/bin/python" > "$NPO_PREP/old-freeze.txt"
uv venv --python "$NPO_OLD_ENV/bin/python" "$NPO_ENV"
uv pip sync --python "$NPO_ENV/bin/python" "$NPO_PREP/old-freeze.txt"
```

`NPO_ENV` 应是一个全新的目录。这里重建包环境，不直接复制含有绝对路径的 venv 文件夹。若旧 torch 的 `+cuXXX` 版本来自专用索引，sync 时使用原来的 PyTorch 索引；本地编译 wheel 同样需要保留原安装来源。freeze 不保存驱动、系统 CUDA Toolkit 或全部编译参数。

从固定官方 requirements 生成一份独立安装清单，仅替换三个 GPU 依赖为旧环境的实际版本：

```bash
"$NPO_OLD_ENV/bin/python" - "$NPO_UPSTREAM/requirements.txt" "$NPO_PREP/requirements-6000d.txt" <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys

hardware = {name: version(name) for name in ('torch', 'bitsandbytes', 'deepspeed')}
lines = []
for line in Path(sys.argv[1]).read_text().splitlines():
    name = line.split('==', 1)[0].strip()
    lines.append(f'{name}=={hardware[name]}' if name in hardware else line)
lines.append('lm-eval==0.4.11')  # 固定提交 setup.py 的 evaluator 依赖
Path(sys.argv[2]).write_text('\n'.join(lines) + '\n')
print(hardware)
PY

uv pip install --python "$NPO_ENV/bin/python" -r "$NPO_PREP/requirements-6000d.txt" --dry-run
uv pip install --python "$NPO_ENV/bin/python" -r "$NPO_PREP/requirements-6000d.txt"
uv pip check --python "$NPO_ENV/bin/python"
uv pip freeze --python "$NPO_ENV/bin/python" > "$NPO_PREP/final-freeze.txt"
```

检查 dry-run 是否保留了三个 GPU 依赖的选定版本。不要直接再安装未调整的官方 requirements 或 `.[lm-eval]`，它们会重新要求 torch=2.4.1。旧环境复制来的其他包如果与新依赖冲突，应在新环境里处理并记录；以 `uv pip check` 和实际导入结果为准。[uv 环境操作](https://docs.astral.sh/uv/pip/environments/)

## 运行入口

在服务器 PALU_demo 根目录运行：

```bash
bash scripts/unlearn/tofu/train_tofu_npo_official.sh \
  --upstream-dir "$NPO_UPSTREAM" \
  --python "$NPO_ENV/bin/python" \
  --retain-logs /path/to/retain95/TOFU_EVAL.json \
  --output-dir /path/to/results/npo-official-6000d/seed-0 \
  --gpu 0,1,2,3 --seed 0 --dry-run
```

先打印命令，再将 `--dry-run` 换成 `--check-only`，检查固定提交、依赖、可见 GPU 以及官方 train/eval 入口的导入和 Hydra 配置。两种检查都不启动模型训练、不创建结果目录。`--check-only` 的通过不表示 CUDA kernel、优化器或多卡通信已通过运行验证。去掉该选项才会进行完整训练和评估。

默认四卡、SDPA、每卡 batch=4、梯度累积=2、有效 batch=32。官方脚本是两卡、每卡 batch=4、梯度累积=4，默认 FlashAttention-2。有效 batch 相同不保证梯度累积、尾部样本处理或浮点结果相同。`--gpu 0,1` 自动使用累积=4；仍需显式 `--attention flash_attention_2` 才使用该后端。

其余设置来自官方配置：普通 NPO 随机采样，Llama-3.1-8B-Instruct Full、forget05/retain95、10 epochs、lr=1e-5、warmup_epochs=1、beta=0.1、alpha=gamma=1。沿用官方 DeepSpeed JSON，不补写 gradient_clipping。数据来自 `locuslab/TOFU`，tokenizer 沿用官方的 Meta Llama 配置与模板，需要相应访问权限或已有缓存。

`--retain-logs` 必须指向匹配 Llama-3.1-8B Retain95 的原始 `TOFU_EVAL.json`，含有效的逐样本 forget_truth_ratio。官方默认 privleak 还读取 mia_min_k，应使用与该提交指标匹配的完整参考日志。可在官方目录用该环境执行 `python setup_data.py --eval_logs` 下载官方参考日志；数据仓库会变化，需记录实际来源。文件名和结构检查不能证明 reference 模型、split、数据 revision 正确。

## 结果与检查范围

训练完成后临时保存模型，所选第一张卡独立评估。默认验证最终 FQ/MU 的汇总及原始记录后清理本次模型权重；`--keep-model` 保留。失败时保留权重和日志。不会覆盖已有结果目录。

```text
seed-0/
  baseline_audit/
    run_contract.json          # 官方 SHA、命令、环境差异、GPU、batch、参考日志 SHA256
    resolved_config.yaml       # 官方训练配置
    resolved_eval_config.yaml  # 官方评估配置
    train.log
    eval.log
    status.json                # 状态、实际 global_step/epoch、汇总结果
  trainer_state.json
  evals/
    TOFU_EVAL.json
    TOFU_SUMMARY.json
```

评估保持该提交的默认指标，包含 FQ、MU、privleak、extraction strength 等；默认没有启用 Exact Memorization 和 gibberish，因此不会自动生成此前四指标图的全部字段。代码提交固定不等于模型/数据 Hub revision 也已经固定。

当前交付为独立分支、启动器和离线检查；6000D 上的新依赖组合、完整训练、首步梯度与数值复现尚未验证。后续先检查实际 microbatch、梯度缩放、裁剪和更新次数，再比较最终 FQ/MU；官方路径通过后才接回 R 采样器。
