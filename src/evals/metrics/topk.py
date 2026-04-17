# src/evals/metrics/topk.py
import json
import logging
import os
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evals.metrics.base import unlearning_metric

logger = logging.getLogger("metrics")


@unlearning_metric(name="log_topk_tokens")
def log_topk_tokens(
    model,
    *,
    data,
    collators=None,
    tokenizer,
    positions: List[int],
    k: int = 5,
    batch_size: int = 1,
    split: str = "forget",
    output_filename: str = None,
    output_path: str = None,
    metric_name: str = None,
    output_dir: str = None,
    device: str = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    取指定 positions 的 top-K 候选词并写文件。
    positions/labels 是按 logits 维度来（logits[:, t] -> 预测 token t+1）。

    Args:
        output_filename: 输出文件名（如果提供了 output_dir，会使用这个文件名）
        output_path: 完整的输出路径（如果提供，优先使用这个）
        metric_name: metric 名称（用于构建默认输出路径）
    """
    device = device or model.device
    model.eval()

    # 选数据集 & collate
    ds = data[split] if isinstance(data, dict) else data
    collate_fn = None
    if isinstance(collators, dict):
        # 取和 split 名对应的 collate_fn，如果没有就 None
        collate_fn = collators.get(split, None)
    elif collators is not None:
        collate_fn = collators

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    records = []

    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(loader, desc="log_topk_tokens")):
            # 只保留模型需要的键
            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ("input_ids", "attention_mask")
            }
            logits = model(**inputs).logits  # [B, T, V]
            B, T, V = logits.shape
            max_pos = max(positions) if positions else 0
            if max_pos >= T:
                # 如果给的 pos 超出长度，截掉
                valid_positions = [p for p in positions if p < T]
            else:
                valid_positions = positions

            for b in range(B):
                entry = {
                    "sample_index": (
                        int(batch.get("index", [batch_id])[b])
                        if isinstance(batch.get("index"), torch.Tensor)
                        else batch_id
                    ),
                    "topk": {},
                }
                for pos in valid_positions:
                    values, idxs = torch.topk(logits[b, pos], k=min(k, V))
                    tokens = [tokenizer.decode([i]) for i in idxs.tolist()]
                    entry["topk"][str(pos)] = {
                        "token_ids": idxs.tolist(),
                        "tokens": tokens,
                        "logits": values.tolist(),
                    }
                records.append(entry)

    # 确定输出路径
    # 优先使用 output_path（完整路径）
    # 其次使用 output_dir + output_filename
    # 最后使用 metric_name 构建默认路径
    final_output_path = output_path
    if not final_output_path:
        # 从 kwargs 或参数中获取 output_dir
        output_dir = output_dir or kwargs.get("output_dir")
        if output_dir and output_filename:
            # 如果提供了 output_dir 和 output_filename，组合使用
            final_output_path = os.path.join(output_dir, output_filename)
        elif output_filename:
            # 只有 output_filename，使用当前工作目录
            final_output_path = (
                output_filename
                if os.path.isabs(output_filename)
                else os.path.join(os.getcwd(), output_filename)
            )
        elif metric_name:
            # 使用 metric_name 构建默认文件名
            if output_dir:
                final_output_path = os.path.join(
                    output_dir, f"{metric_name}_topk_tokens.json"
                )
            else:
                final_output_path = f"{metric_name}_topk_tokens.json"

    if final_output_path:
        os.makedirs(
            (
                os.path.dirname(final_output_path)
                if os.path.dirname(final_output_path)
                else "."
            ),
            exist_ok=True,
        )
        with open(final_output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info(f"Top-K tokens saved to {final_output_path}")

    return {"agg_value": len(records), "topk_records": records}
