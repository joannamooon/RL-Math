from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from cs336_alignment.grpo import (
    aggregate_loss_across_microbatch,
    compute_group_normalized_rewards,
    compute_policy_gradient_loss,
    compute_rollout_rewards,
    get_response_log_probs,
    grpo_train_step,
    tokenize_prompt_and_output,
)

# GRPO adapters: thin re-exports. The real implementation lives in
# cs336_alignment/grpo.py so it ships in the Modal image (tests/ does not).
run_tokenize_prompt_and_output = tokenize_prompt_and_output
run_get_response_log_probs = get_response_log_probs
run_compute_rollout_rewards = compute_rollout_rewards
run_compute_group_normalized_rewards = compute_group_normalized_rewards
run_compute_policy_gradient_loss = compute_policy_gradient_loss
run_aggregate_loss_across_microbatch = aggregate_loss_across_microbatch
run_grpo_train_step = grpo_train_step


"""
The below adapters are used in the optional
RLHF / safety part of the Alignment assignment.
"""


_ALPACA_SFT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n{response}"
)


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).
    """
    examples = []
    with open(dataset_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))

    if shuffle:
        rng = random.Random(0)
        rng.shuffle(examples)

    eos_id = tokenizer.eos_token_id

    all_ids: list[int] = []
    for example in examples:
        text = _ALPACA_SFT_TEMPLATE.format(
            instruction=example["prompt"],
            response=example["response"],
        )
        ids = tokenizer(text)["input_ids"]
        all_ids.extend(ids)
        if eos_id is not None:
            all_ids.append(eos_id)

    num_examples = (len(all_ids) - 1) // seq_length

    input_ids_list = []
    labels_list = []
    for i in range(num_examples):
        start = i * seq_length
        chunk = all_ids[start : start + seq_length + 1]
        input_ids_list.append(torch.tensor(chunk[:-1], dtype=torch.long))
        labels_list.append(torch.tensor(chunk[1:], dtype=torch.long))

    class _PackedSFTDataset(Dataset):
        def __len__(self) -> int:
            return len(input_ids_list)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {
                "input_ids": input_ids_list[idx],
                "labels": labels_list[idx],
            }

    return _PackedSFTDataset()


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.
    """
    from torch.utils.data import DataLoader

    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        drop_last=False,
    )


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D').
    """
    match = re.search(r"correct answer is\s*\**\(?([A-D])\)?\b", model_output, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"\b([A-D])\b", model_output)
    if match:
        return match.group(1).upper()

    return None


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.
    """
    matches = re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?", model_output)
    if not matches:
        return None

    last = matches[-1].replace(",", "").replace("$", "")
    return last


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.
    """

    def sequence_log_prob(model: torch.nn.Module, prompt_str: str, response_str: str) -> torch.Tensor:
        device = next(model.parameters()).device
        tokenized = run_tokenize_prompt_and_output(
            prompt_strs=[prompt_str],
            output_strs=[response_str],
            tokenizer=tokenizer,
        )
        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        mask = tokenized["response_mask"].to(device).to(torch.float32)

        log_probs = run_get_response_log_probs(
            model=model,
            input_ids=input_ids,
            labels=labels,
            return_token_entropy=False,
        )["log_probs"]

        return (log_probs * mask).sum()

    policy_chosen_logp = sequence_log_prob(lm, prompt, response_chosen)
    policy_rejected_logp = sequence_log_prob(lm, prompt, response_rejected)

    with torch.no_grad():
        ref_chosen_logp = sequence_log_prob(lm_ref, prompt, response_chosen)
        ref_rejected_logp = sequence_log_prob(lm_ref, prompt, response_rejected)

    ref_chosen_logp = ref_chosen_logp.to(policy_chosen_logp.device)
    ref_rejected_logp = ref_rejected_logp.to(policy_chosen_logp.device)

    logits = beta * (
        (policy_chosen_logp - policy_rejected_logp) - (ref_chosen_logp - ref_rejected_logp)
    )
    loss = -F.logsigmoid(logits)

    return loss
