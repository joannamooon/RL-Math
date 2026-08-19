"""
GRPO implementation: tokenization, log-probs, rewards, advantages, policy-gradient
loss (on- and off-policy variants), and the full training step.

This is the "real" library code (shipped in the Modal image via cs336_alignment/).
tests/adapters.py re-exports these under the run_* names the test suite expects.
"""

from __future__ import annotations

from typing import Callable, Literal

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask aligned with
    labels that is 1 for response tokens and 0 for other tokens (prompt or padding).
    """
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0

    prompt_ids_list = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompt_strs]
    output_ids_list = [tokenizer(o, add_special_tokens=False)["input_ids"] for o in output_strs]

    full_ids_list = [p + o for p, o in zip(prompt_ids_list, output_ids_list)]
    mask_list = [[0] * len(p) + [1] * len(o) for p, o in zip(prompt_ids_list, output_ids_list)]

    max_len = max(len(ids) for ids in full_ids_list)

    input_ids_rows = []
    labels_rows = []
    mask_rows = []
    for full_ids, mask in zip(full_ids_list, mask_list):
        pad_len = max_len - len(full_ids)
        full_padded = full_ids + [pad_token_id] * pad_len
        mask_padded = mask + [0] * pad_len

        input_ids_rows.append(full_padded[:-1])
        labels_rows.append(full_padded[1:])
        mask_rows.append(mask_padded[1:])

    return {
        "input_ids": torch.tensor(input_ids_rows, dtype=torch.long),
        "labels": torch.tensor(labels_rows, dtype=torch.long),
        "response_mask": torch.tensor(mask_rows, dtype=torch.long),
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities (given the previous tokens)
    from a causal language model, and optionally the entropy of the model's
    next-token distribution.
    """
    logits = model(input_ids).logits
    log_probs_full = torch.log_softmax(logits.float(), dim=-1)
    log_probs = torch.gather(log_probs_full, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    result: dict[str, torch.Tensor] = {"log_probs": log_probs}

    if return_token_entropy:
        probs_full = log_probs_full.exp()
        token_entropy = -(probs_full * log_probs_full).sum(dim=-1)
        result["token_entropy"] = token_entropy

    return result


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rewards for a list of rollout responses, along with metadata for
    the reward components.
    """
    reward_dicts = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths)
    ]

    raw_rewards = torch.tensor([d["reward"] for d in reward_dicts], dtype=torch.float32)
    format_rewards = torch.tensor([d["format_reward"] for d in reward_dicts], dtype=torch.float32)
    answer_rewards = torch.tensor([d["answer_reward"] for d in reward_dicts], dtype=torch.float32)

    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "mean_format_reward": format_rewards.mean().item(),
        "mean_answer_reward": answer_rewards.mean().item(),
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
    }

    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages by applying the requested baseline and normalization
    within each group.
    """
    groups = raw_rewards.view(-1, group_size)
    group_mean = groups.mean(dim=1, keepdim=True)

    if baseline == "mean":
        centered = groups - group_mean
    elif baseline == "none":
        centered = groups
    else:
        raise NotImplementedError(f"Unsupported baseline: {baseline}")

    if advantage_normalizer == "std":
        group_std = groups.std(dim=1, keepdim=True)
        normalized = centered / (group_std + advantage_eps)
    elif advantage_normalizer == "none":
        normalized = centered
    elif advantage_normalizer == "mean":
        normalized = centered / (group_mean + advantage_eps)
    else:
        raise NotImplementedError(f"Unsupported advantage_normalizer: {advantage_normalizer}")

    advantages = normalized.view(-1)

    metadata = {
        "mean_raw_reward": raw_rewards.mean().item(),
        "std_raw_reward": raw_rewards.std().item(),
        "max_raw_reward": raw_rewards.max().item(),
        "min_raw_reward": raw_rewards.min().item(),
        "mean_advantage": advantages.mean().item(),
        "std_advantage": advantages.std().item(),
    }

    return advantages, metadata


def _grpo_clip_loss(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * advantages
    surrogate = torch.min(unclipped, clipped)
    is_clipped = clipped < unclipped
    return surrogate, is_clipped


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy-gradient loss at every token, where
    raw_rewards_or_advantages is either the raw reward or an
    already-normalized advantage.
    """
    advantages = raw_rewards_or_advantages.reshape(-1, 1).to(policy_log_probs.dtype)
    metadata: dict[str, torch.Tensor] = {}

    if importance_reweighting_method == "none":
        per_token_policy_gradient_loss = -(advantages * policy_log_probs)

    elif importance_reweighting_method == "noclip":
        if old_log_probs is None:
            raise ValueError("old_log_probs is required when importance_reweighting_method != 'none'.")
        ratio = torch.exp(policy_log_probs - old_log_probs)
        per_token_policy_gradient_loss = -(ratio * advantages)
        metadata["mean_importance_ratio"] = ratio.mean().detach()

    elif importance_reweighting_method == "grpo":
        if old_log_probs is None or cliprange is None:
            raise ValueError("old_log_probs and cliprange are required for importance_reweighting_method='grpo'.")
        ratio = torch.exp(policy_log_probs - old_log_probs)
        surrogate, is_clipped = _grpo_clip_loss(ratio, advantages, cliprange)
        per_token_policy_gradient_loss = -surrogate
        metadata["mean_importance_ratio"] = ratio.mean().detach()
        metadata["clip_fraction"] = is_clipped.float().mean().detach()

    elif importance_reweighting_method == "gspo":
        if old_log_probs is None or cliprange is None or response_mask is None:
            raise ValueError(
                "old_log_probs, cliprange, and response_mask are required for "
                "importance_reweighting_method='gspo'."
            )
        log_ratio = policy_log_probs - old_log_probs
        mask = response_mask.to(log_ratio.dtype)
        token_counts = mask.sum(dim=1).clamp(min=1.0)
        seq_log_ratio = (log_ratio * mask).sum(dim=1) / token_counts
        seq_ratio = torch.exp(seq_log_ratio)  # shape (batch_size,)

        seq_advantages = advantages.reshape(-1)
        surrogate, is_clipped = _grpo_clip_loss(seq_ratio, seq_advantages, cliprange)
        per_seq_loss = -surrogate  # shape (batch_size,)

        per_token_policy_gradient_loss = per_seq_loss.unsqueeze(1).expand_as(policy_log_probs)
        metadata["mean_importance_ratio"] = seq_ratio.mean().detach()
        metadata["clip_fraction"] = is_clipped.float().mean().detach()

    else:
        raise NotImplementedError(f"Unsupported importance_reweighting_method: {importance_reweighting_method}")

    return per_token_policy_gradient_loss, metadata


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Aggregate the per-token policy-gradient loss according to the response
    mask and loss-normalization strategy.
    """
    mask = mask.to(per_token_policy_gradient_loss.dtype)

    if loss_normalization == "sequence":
        seq_lens = mask.sum(dim=1).clamp(min=1.0)
        seq_losses = (per_token_policy_gradient_loss * mask).sum(dim=1) / seq_lens
        loss = seq_losses.mean()
    elif loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError("normalization_constant is required when loss_normalization='constant'.")
        loss = (per_token_policy_gradient_loss * mask).sum() / normalization_constant
    else:
        raise NotImplementedError(f"Unsupported loss_normalization: {loss_normalization}")

    return loss


def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Execute forward-and-backward passes, with gradient_accumulation_steps
    microbatches.
    """
    device = next(model.parameters()).device

    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
    )
    advantages, adv_metadata = compute_group_normalized_rewards(
        raw_rewards=raw_rewards,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )
    advantages = advantages.to(device)

    tokenized = tokenize_prompt_and_output(
        prompt_strs=repeated_prompts,
        output_strs=rollout_responses,
        tokenizer=tokenizer,
    )
    input_ids = tokenized["input_ids"].to(device)
    labels = tokenized["labels"].to(device)
    response_mask = tokenized["response_mask"].to(device)

    if old_log_probs is not None:
        old_log_probs = old_log_probs.to(device)

    batch_size = input_ids.shape[0]
    microbatch_size = batch_size // gradient_accumulation_steps
    if microbatch_size == 0:
        microbatch_size = batch_size

    total_loss = torch.zeros((), device=device)
    total_entropy = torch.zeros((), device=device)
    total_entropy_tokens = torch.zeros((), device=device)
    last_step_metadata: dict[str, torch.Tensor] = {}

    for start in range(0, batch_size, microbatch_size):
        end = min(start + microbatch_size, batch_size)
        mb_input_ids = input_ids[start:end]
        mb_labels = labels[start:end]
        mb_mask = response_mask[start:end]
        mb_advantages = advantages[start:end]
        mb_old_log_probs = old_log_probs[start:end] if old_log_probs is not None else None

        log_probs_out = get_response_log_probs(
            model=model,
            input_ids=mb_input_ids,
            labels=mb_labels,
            return_token_entropy=True,
        )
        policy_log_probs = log_probs_out["log_probs"]
        token_entropy = log_probs_out["token_entropy"]

        per_token_loss, step_metadata = compute_policy_gradient_loss(
            raw_rewards_or_advantages=mb_advantages,
            policy_log_probs=policy_log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=mb_old_log_probs,
            cliprange=cliprange,
            response_mask=mb_mask,
        )
        last_step_metadata = step_metadata

        if loss_normalization == "sequence":
            mb_loss = aggregate_loss_across_microbatch(
                per_token_policy_gradient_loss=per_token_loss,
                mask=mb_mask,
                loss_normalization="sequence",
            )
            scale = (end - start) / batch_size
        else:
            mb_loss = aggregate_loss_across_microbatch(
                per_token_policy_gradient_loss=per_token_loss,
                mask=mb_mask,
                loss_normalization="constant",
                normalization_constant=normalization_constant,
            )
            scale = 1.0

        scaled_loss = mb_loss * scale
        scaled_loss.backward()
        total_loss = total_loss + scaled_loss.detach()

        mask_f = mb_mask.to(token_entropy.dtype)
        total_entropy = total_entropy + (token_entropy * mask_f).sum().detach()
        total_entropy_tokens = total_entropy_tokens + mask_f.sum().detach()

    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    mean_entropy = (
        (total_entropy / total_entropy_tokens) if total_entropy_tokens.item() > 0 else torch.zeros((), device=device)
    )

    metadata: dict[str, torch.Tensor | float] = {
        "grad_norm": grad_norm.detach(),
        "token_entropy": mean_entropy.detach(),
        **reward_metadata,
        **adv_metadata,
        **{k: v for k, v in last_step_metadata.items()},
    }

    return total_loss, metadata
