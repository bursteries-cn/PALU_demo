import torch
import torch.nn.functional as F

from trainer.unlearn.grad_diff import GradDiff


def _group_sensitive_indices_into_intervals(indices: torch.Tensor) -> list:
    lst = indices.tolist()
    if isinstance(lst, int):
        lst = [lst]
    if len(lst) == 0:
        return []
    if len(lst) == 1:
        return [lst]

    intervals = []
    current = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] > 1:
            intervals.append(current)
            current = [lst[i]]
        else:
            current.append(lst[i])
    intervals.append(current)
    return intervals


def _select_from_interval(
    interval_positions: list,
    k: int,
) -> list:
    n = len(interval_positions)
    if n == 0:
        return []

    num_keep = min(k, n)

    return interval_positions[:num_keep]


class PALU(GradDiff):
    def __init__(
        self,
        top_k=1,
        first_n=-1,
        target_mode="mean",
        non_sensitive_constraint_type="kl",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.top_k = top_k
        self.num_sensitive_tokens = first_n

        self.target_mode = target_mode
        self.non_sensitive_constraint_type = non_sensitive_constraint_type

        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]

        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
        }

        forget_outputs = model(**forget_inputs)
        logits = forget_outputs.logits  # [B, T, V]

        forget_labels = inputs["forget"]["labels"]  # [B, T]
        shifted_logits = logits[..., :-1, :].contiguous()  # [B, T-1, V]
        shifted_labels = forget_labels[..., 1:].contiguous()  # [B, T-1]

        logits_flat = shifted_logits.view(-1, shifted_logits.size(-1))  # [N, V]
        _, V = logits_flat.size()
        k = min(self.top_k, V)

        B, T_shifted = shifted_labels.size()
        N = B * T_shifted
        forget_labels_flat = shifted_labels.reshape(-1)  # [N]

        question_mask = forget_labels_flat == -100  # [N]
        non_sensitive_mask = forget_labels_flat == -200  # [N]
        sensitive_mask = forget_labels_flat >= 0  # [N]

        eos = self.tokenizer.eos_token_id
        if eos is not None:
            sensitive_mask = sensitive_mask & (forget_labels_flat != eos)

        if self.num_sensitive_tokens > 0:
            refined_sensitive_mask = torch.zeros_like(sensitive_mask)
            device = sensitive_mask.device

            for b in range(B):
                batch_labels = shifted_labels[b]
                sensitive_indices = torch.nonzero(
                    (
                        (batch_labels >= 0) & (batch_labels != eos)
                        if eos is not None
                        else (batch_labels >= 0)
                    ),
                    as_tuple=False,
                ).squeeze(-1)

                if sensitive_indices.numel() > 0:
                    if sensitive_indices.dim() == 0:
                        sensitive_indices = sensitive_indices.unsqueeze(0)

                    intervals = _group_sensitive_indices_into_intervals(
                        sensitive_indices
                    )
                    k_interval = self.num_sensitive_tokens

                    all_keep_positions = []
                    for interval in intervals:
                        keep_in_interval = _select_from_interval(interval, k_interval)
                        all_keep_positions.extend(keep_in_interval)

                    if all_keep_positions:
                        keep_positions = torch.tensor(
                            all_keep_positions, device=device, dtype=torch.long
                        )
                        flat_positions = keep_positions + b * T_shifted
                        refined_sensitive_mask[flat_positions] = True

            sensitive_mask = refined_sensitive_mask

        # ref_model
        with torch.no_grad():
            ref_outputs = self.ref_model(**forget_inputs)
        ref_logits = ref_outputs.logits[..., :-1, :].contiguous()
        ref_logits_flat = ref_logits.view(-1, V)
        _, ref_topk_indices = torch.topk(ref_logits_flat, k, dim=-1)

        batch_indices = (
            torch.arange(N, device=logits_flat.device).unsqueeze(1).expand(-1, k)
        )
        selected_logits = logits_flat[batch_indices, ref_topk_indices]

        if self.target_mode == "mean":
            averageLogits = logits_flat.mean(dim=-1, keepdim=True).expand(-1, k)
            target_logits = averageLogits
        else:
            raise ValueError(
                f"Unknown target_mode: {self.target_mode}. Supported modes: mean"
            )

        forget_loss = (selected_logits - target_logits) ** 2
        forget_loss = forget_loss.mean(dim=-1)
        sensitive_forget_loss = (
            forget_loss * sensitive_mask
        ).sum() / sensitive_mask.sum().clamp(min=1)

        if non_sensitive_mask.sum() > 0:
            if self.non_sensitive_constraint_type == "kl":
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                ref_log_probs_flat = ref_log_probs.view(-1, V)
                current_log_probs = F.log_softmax(shifted_logits, dim=-1)
                current_log_probs_flat = current_log_probs.view(-1, V)
                kl_per_token = torch.sum(
                    torch.exp(current_log_probs_flat)
                    * (current_log_probs_flat - ref_log_probs_flat),
                    dim=-1,
                )
                non_sensitive_constraint_loss = (
                    kl_per_token * non_sensitive_mask
                ).sum() / non_sensitive_mask.sum().clamp(min=1)
            elif self.non_sensitive_constraint_type == "nll":
                ref_predicted_labels = torch.argmax(ref_logits_flat, dim=-1)
                current_log_probs = F.log_softmax(shifted_logits, dim=-1)
                current_log_probs_flat = current_log_probs.view(-1, V)
                nll_per_token = -current_log_probs_flat[
                    torch.arange(N, device=logits_flat.device), ref_predicted_labels
                ]
                non_sensitive_constraint_loss = (
                    nll_per_token * non_sensitive_mask
                ).sum() / non_sensitive_mask.sum().clamp(min=1)
            else:
                raise ValueError(
                    f"Unknown non_sensitive_constraint_type: "
                    f"{self.non_sensitive_constraint_type}. Supported: kl, nll"
                )
        else:
            non_sensitive_constraint_loss = torch.tensor(
                0.0, device=forget_outputs.logits.device
            )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = (
            self.gamma * sensitive_forget_loss
            + self.alpha * retain_loss
            + self.alpha * non_sensitive_constraint_loss
        )
        return (loss, forget_outputs) if return_outputs else loss
