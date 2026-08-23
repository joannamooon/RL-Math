import torch 
import torch.nn 
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Callable, Literal
from torch.optim import Optimizer
import wandb
import torch.distributed as dist 

from vllm_utils import VLLMServer
from checkpoint import get_model_and_tokenizer

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    prompt_and_output_lens = []
    input_ids_list = []
    labels_list = []
    response_mask_list = []
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)
        full_ids = prompt_ids + output_ids
        prompt_and_output_lens.append(len(full_ids))
        input_id = torch.tensor(full_ids[:-1])
        label = torch.tensor(full_ids[1:])
        loss_mask = torch.tensor([0] * len(prompt_ids) + [1] * len(output_ids))[1:]
        input_ids_list.append(input_id)
        labels_list.append(label)
        response_mask_list.append(loss_mask)
    
    pad_id = tokenizer.pad_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=pad_id)
    response_mask = torch.nn.utils.rnn.pad_sequence(response_mask_list, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids, 
        "labels": labels, 
        "response_mask": response_mask
    }

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    device = model.device
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    logits = model(input_ids).logits

    log_probs_full = torch.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs_full, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    result = {}
    result["log_probs"] = log_probs
    if return_token_entropy:
        probs_full = torch.exp(log_probs_full)
        token_entropy = - (probs_full * log_probs_full).sum(dim=-1)
        result["token_entropy"] = token_entropy
    
    return result 

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_rewards = []
    total_reward = 0 
    total_format_reward = 0 
    total_answer_reward = 0 
    for response, truth in zip(rollout_responses, repeated_ground_truths):
        data = reward_fn(response, truth)
        raw_rewards.append(data["reward"])
        total_reward += data["reward"]
        total_format_reward += data["format_reward"]
        total_answer_reward += data["answer_reward"]
    mean_reward = total_reward / len(rollout_responses)
    mean_format_reward = total_format_reward / len(rollout_responses)
    mean_answer_reward = total_answer_reward / len(rollout_responses)
    metadata = {}
    metadata["mean_reward"] = mean_reward
    metadata["mean_format_reward"] = mean_format_reward
    metadata["mean_answer_reward"] = mean_answer_reward
    raw_rewards = torch.tensor(raw_rewards)
    return (raw_rewards, metadata)

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
): 
    n_prompts_per_rollout_batch = raw_rewards.shape[0] // group_size
    grouped_rewards = torch.reshape(raw_rewards, (n_prompts_per_rollout_batch, group_size))

    if baseline == "mean":
        group_baseline = grouped_rewards.mean(dim=1, keepdim=True)
        grouped_rewards -= group_baseline
    
    if advantage_normalizer == "std":
        group_std = grouped_rewards.std(dim=1, keepdim=True)
        grouped_rewards /= (group_std + advantage_eps)
    elif advantage_normalizer == "none":
        pass
    else:
        group_baseline = grouped_rewards.mean(dim=-1, keepdim=True)
        group_rewards /= (group_baseline + advantage_eps)
    
    advantages = torch.reshape(grouped_rewards, (raw_rewards.shape[0], ))
    metadata = {}
    metadata["mean"] = advantages.mean()
    metadata["std"] = advantages.std()
    metadata["max"] = advantages.max()
    metadata["min"] = advantages.min()
    return (advantages, metadata)

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    advantages = raw_rewards_or_advantages
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1) # [batch_size] => [batch_size, 1]

    if importance_reweighting_method == "none":
        per_token_loss = -advantages * policy_log_probs
    elif importance_reweighting_method == "noclip":
        w = torch.exp(policy_log_probs - old_log_probs)
        per_token_loss = -advantages * w
    elif importance_reweighting_method == "grpo":
        w = torch.exp(policy_log_probs - old_log_probs)
        per_token_loss = -torch.minimum(advantages * w, advantages * torch.clamp(w, 1 - cliprange, 1 + cliprange))
    else:
        w = torch.exp(policy_log_probs - old_log_probs)
        log_ratio = log_ratio * response_mask

        seq_len = response_mask.sum(dim=-1, keepdim=True)
        log_s = log_ratio.sum(dim=-1, keepdim=True) / seq_len 
        s = torch.exp(log_s)
        clipped_s = torch.clamp(s, 1 - cliprange, 1 + cliprange)
        per_token_loss = -torch.minimum(advantages * s, advantages * clipped_s)
        
    metadata = {}
    return (per_token_loss, metadata)


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    masked_loss = per_token_policy_gradient_loss * mask 
    response_lengths = mask.sum(dim=-1) # (batch_size,)
    if loss_normalization == "sequence":
        normalized_loss = masked_loss.sum(dim=-1) / response_lengths #(batch_size,)
    else:
        normalized_loss = masked_loss.sum(dim=-1) / normalization_constant
    return normalized_loss.mean()

def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    total_loss = torch.tensor(0.0, device=model.device)
    tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    response_mask = tokenized["response_mask"]

    microbatch_size = len(input_ids) // gradient_accumulation_steps

    raw_rewards, reward_metadata = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    advantages, adv_metadata = compute_group_normalized_rewards(raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer)
    
    if old_log_probs is None:
        with torch.no_grad():
            old_log_probs_list = []
            for i in range(0, len(input_ids), microbatch_size):
                microbatch_input_ids = input_ids[i:i+microbatch_size]
                microbatch_labels = labels[i:i+microbatch_size]
                old_log_probs_dict = get_response_log_probs(model, microbatch_input_ids, microbatch_labels, False)
                old_log_probs_list.append(old_log_probs_dict["log_probs"])
            old_log_probs = torch.cat(old_log_probs_list, dim=0)

    for i in range(0, len(input_ids), microbatch_size):
        microbatch_input_ids = input_ids[i:i+microbatch_size]
        microbatch_labels = labels[i:i+microbatch_size]
        microbatch_response_mask = response_mask[i:i+microbatch_size]
        microbatch_advantages = advantages[i:i+microbatch_size]
        microbatch_old_log_probs = old_log_probs[i:i+microbatch_size]

        log_probs_dict = get_response_log_probs(model, microbatch_input_ids, microbatch_labels, True)
        log_probs = log_probs_dict["log_probs"]
        token_entropy = log_probs_dict["token_entropy"]

        
        per_token_policy_gradient_loss, _ = compute_policy_gradient_loss(
            microbatch_advantages, log_probs, importance_reweighting_method, microbatch_old_log_probs, cliprange, microbatch_response_mask
        )

        loss = aggregate_loss_across_microbatch(per_token_policy_gradient_loss, microbatch_response_mask, loss_normalization, normalization_constant)
        loss *= (len(microbatch_input_ids) / len(input_ids))

        # adds loss.backward to param.grad every time 
        loss.backward()

    total_loss += loss
    if max_grad_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # once .grad is accumulated, step and zero out 
    optimizer.step()
    optimizer.zero_grad()

    metadata = {**reward_metadata, **adv_metadata}
    return loss, metadata

