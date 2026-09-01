# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

import logging
import math
import os
from typing import Any, Dict, List, Optional, Union

from torch.utils.data import Dataset
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

logger = logging.getLogger(__name__)

# KS p-values can underflow to exactly 0; floor them so the log stays on a readable scale.
MIN_PVALUE = 1e-30


def add_neg_log_pvalue_metrics(eval_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Add -log10 versions of p-value metrics, which span too many orders of
    magnitude to be readable on the linear axes of a logging dashboard."""
    for name in ("forget_quality",):
        pvalue = eval_metrics.get(name)
        if isinstance(pvalue, (int, float)) and not isinstance(pvalue, bool):
            eval_metrics[f"{name}_neg_log10"] = -math.log10(max(pvalue, MIN_PVALUE))
    return eval_metrics


class FinetuneTrainer(Trainer):
    def __init__(self, evaluators=None, template_args=None, *args, **kwargs):
        self.evaluators = evaluators
        self.template_args = template_args
        super().__init__(*args, **kwargs)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        # Run a custom evaluator and save results
        if self.evaluators:
            if self.accelerator.is_local_main_process:
                eval_metrics = {}
                if self.accelerator.num_processes == 1:
                    run_dir = self._get_output_dir(trial=trial)
                    checkpoint_folder = (
                        f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                    )
                    output_dir = os.path.join(run_dir, checkpoint_folder, "evals")
                    os.makedirs(output_dir, exist_ok=True)
                    eval_metrics = {}
                    for _, evaluator in self.evaluators.items():
                        eval_args = {
                            "output_dir": output_dir,
                            "template_args": self.template_args,
                            "model": self.model,
                            "tokenizer": self.tokenizer,
                        }
                        eval_metrics.update(evaluator.evaluate(**eval_args))
                    self.log(add_neg_log_pvalue_metrics(dict(eval_metrics)))
                else:
                    logger.warning(
                        "Custom evaluator can be run with this Trainer only when a single accelerator process is running."
                    )
                return eval_metrics

        if eval_dataset is None:
            return {}
        # Run the default HF Trainer evaluate method when eval dataset is provided
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
