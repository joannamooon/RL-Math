
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable, Literal

import torch
import wandb

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from checkpoint import get_model_and_tokenizer
from drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from grpo import (
    compute_rollout_rewards,
    get_response_log_probs,
    grpo_train_step,
    tokenize_prompt_and_output,
)
from vllm_utils import VLLMServer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TRAIN_PATH = REPO_ROOT / "data" / "gsm8k" / "train.jsonl"
DEFAULT_VAL_PATH = REPO_ROOT / "data" / "gsm8k" / "test.jsonl"
PROMPT_DIR = _SRC / "prompts"

PromptName = Literal["r1_zero", "r1_zero_three_shot", "question_only"]
Baseline = Literal["mean", "none"]
AdvantageNormalizer = Literal["std", "none", "mean"]
LossNormalization = Literal["sequence", "constant"]
ImportanceReweightingMethod = Literal["none", "noclip", "grpo", "gspo"]


def load_gsm8k(path: str | Path) -> list[dict]:
    examples = []
    with open(path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def extract_ground_truth(example: dict) -> str:
    return example["answer"].split("####")[-1].strip()


def load_prompt_template(prompt_name: PromptName) -> str:
    prompt_files = {
        "r1_zero": "r1_zero.prompt",
        "r1_zero_three_shot": "r1_zero_three_shot_gsm8k.prompt",
        "question_only": "question_only.prompt",
    }
    with open(PROMPT_DIR / prompt_files[prompt_name], "r") as f:
        return f.read()


def get_reward_fn(prompt_name: PromptName) -> Callable[[str, str], dict[str, float]]:
    if prompt_name == "question_only":
        return question_only_reward_fn
    return r1_zero_reward_fn


def get_sampling_stop(prompt_name: PromptName) -> tuple[str | None, bool]:
    if prompt_name == "question_only":
        return None, False
    return "</answer>", True


def expand_for_group_size(
    prompts: list[str],
    ground_truths: list[str],
    group_size: int,
) -> tuple[list[str], list[str]]:
    repeated_prompts: list[str] = []
    repeated_ground_truths: list[str] = []
    for prompt, ground_truth in zip(prompts, ground_truths):
        repeated_prompts.extend([prompt] * group_size)
        repeated_ground_truths.extend([ground_truth] * group_size)
    return repeated_prompts, repeated_ground_truths


def train_grpo(
    model_name: str,
    prompt_name: PromptName,
    training_set_path: str | Path,
    validation_set_path: str | Path,
    seed: int,
    wandb_project: str,
    wandb_run_name: str | None = None,
    n_train_examples: int = 6400,
    n_val_examples: int = 1024,
    num_rollout_steps: int = 200,
    learning_rate: float = 1e-5,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    gradient_accumulation_steps: int = 32,
    num_gradient_steps: int = 1,
    sampling_temperature: float = 1.0,
    sampling_max_tokens: int = 512,
    max_grad_norm: float = 1.0,
    val_every: int = 10,
    policy_device: str = "cuda:0",
    vllm_gpu: int = 1,
    baseline: Baseline = "mean",
    advantage_normalizer: AdvantageNormalizer = "std",
    loss_normalization: LossNormalization = "sequence",
    normalization_constant: int | None = None,
    importance_reweighting_method: ImportanceReweightingMethod = "none",
    cliprange: float | None = None,
) -> None:
    if rollout_batch_size % group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size.")
    if num_gradient_steps < 1:
        raise ValueError("num_gradient_steps must be at least 1.")

    prompts_per_step = rollout_batch_size // group_size
    if normalization_constant is None and loss_normalization == "constant":
        normalization_constant = rollout_batch_size

    random.seed(seed)
    torch.manual_seed(seed)

    training_examples = load_gsm8k(training_set_path)[:n_train_examples]
    validation_examples = load_gsm8k(validation_set_path)[:n_val_examples]

    prompt_template = load_prompt_template(prompt_name)
    reward_fn = get_reward_fn(prompt_name)
    stop, include_stop_str_in_output = get_sampling_stop(prompt_name)

    sampling_params = {
        "temperature": sampling_temperature,
        "max_tokens": sampling_max_tokens,
        "n": group_size,
        "seed": seed,
        "stop": stop,
        "include_stop_str_in_output": include_stop_str_in_output,
    }
    val_sampling_params = {**sampling_params, "n": 1}

    model, tokenizer = get_model_and_tokenizer(model_name, device=policy_device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    vllm_server = VLLMServer(model_id=model_name, gpu=vllm_gpu, seed=seed)
    vllm_server.start()
    vllm_server.init_weight_sync(policy_device=policy_device)
    vllm_server.sync_policy_weights(policy=model)

    config = {
        "model_name": model_name,
        "prompt_name": prompt_name,
        "seed": seed,
        "n_train_examples": n_train_examples,
        "n_val_examples": n_val_examples,
        "num_rollout_steps": num_rollout_steps,
        "learning_rate": learning_rate,
        "rollout_batch_size": rollout_batch_size,
        "group_size": group_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_gradient_steps": num_gradient_steps,
        "sampling_temperature": sampling_temperature,
        "sampling_max_tokens": sampling_max_tokens,
        "max_grad_norm": max_grad_norm,
        "baseline": baseline,
        "advantage_normalizer": advantage_normalizer,
        "loss_normalization": loss_normalization,
        "normalization_constant": normalization_constant,
        "importance_reweighting_method": importance_reweighting_method,
        "cliprange": cliprange,
    }

    with wandb.init(project=wandb_project, name=wandb_run_name, config=config) as run:
        for step in range(num_rollout_steps):
            batch_start = step * prompts_per_step
            training_batch = training_examples[batch_start : batch_start + prompts_per_step]
            train_prompts = [
                prompt_template.format(question=example["question"])
                for example in training_batch
            ]
            ground_truths = [extract_ground_truth(example) for example in training_batch]
            repeated_prompts, repeated_ground_truths = expand_for_group_size(
                train_prompts,
                ground_truths,
                group_size,
            )

            vllm_server.sync_policy_weights(policy=model)
            rollout_completions = vllm_server.generate_completions(
                prompts=train_prompts,
                sampling_params=sampling_params,
            )
            rollout_responses = [completion.text for completion in rollout_completions]

            old_log_probs = None
            if num_gradient_steps > 1 or importance_reweighting_method != "none":
                tokenized = tokenize_prompt_and_output(
                    repeated_prompts,
                    rollout_responses,
                    tokenizer,
                )
                with torch.no_grad():
                    old_log_probs = get_response_log_probs(
                        model,
                        tokenized["input_ids"],
                        tokenized["labels"],
                        return_token_entropy=False,
                    )["log_probs"]

            loss = None
            metadata: dict[str, float] = {}
            for _ in range(num_gradient_steps):
                loss, metadata = grpo_train_step(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    max_grad_norm=max_grad_norm,
                    reward_fn=reward_fn,
                    repeated_prompts=repeated_prompts,
                    rollout_responses=rollout_responses,
                    repeated_ground_truths=repeated_ground_truths,
                    group_size=group_size,
                    baseline=baseline,
                    advantage_normalizer=advantage_normalizer,
                    loss_normalization=loss_normalization,
                    normalization_constant=normalization_constant,
                    importance_reweighting_method=importance_reweighting_method,
                    old_log_probs=old_log_probs,
                    cliprange=cliprange,
                )

            assert loss is not None
            run.log(
                {
                    "train/loss": loss.item(),
                    "train/mean_reward": metadata["mean_reward"],
                    "train/mean_format_reward": metadata["mean_format_reward"],
                    "train/mean_answer_reward": metadata["mean_answer_reward"],
                    "train/advantage_mean": float(metadata["mean"]),
                    "train/advantage_std": float(metadata["std"]),
                },
                step=step,
            )

            if step % val_every == 0:
                val_prompts = [
                    prompt_template.format(question=example["question"])
                    for example in validation_examples
                ]
                val_ground_truths = [
                    extract_ground_truth(example) for example in validation_examples
                ]
                vllm_server.sync_policy_weights(policy=model)
                val_completions = vllm_server.generate_completions(
                    prompts=val_prompts,
                    sampling_params=val_sampling_params,
                    batch_size=32,
                )
                val_responses = [completion.text for completion in val_completions]
                _, val_metadata = compute_rollout_rewards(
                    reward_fn=reward_fn,
                    rollout_responses=val_responses,
                    repeated_ground_truths=val_ground_truths,
                )
                run.log(
                    {
                        "val/mean_reward": val_metadata["mean_reward"],
                        "val/mean_format_reward": val_metadata["mean_format_reward"],
                        "val/mean_answer_reward": val_metadata["mean_answer_reward"],
                    },
                    step=step,
                )

    vllm_server.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a model with GRPO on GSM8K.")
    parser.add_argument("--model-id", default="allenai/OLMo-2-0425-1B")
    parser.add_argument(
        "--prompt",
        choices=["r1_zero", "r1_zero_three_shot", "question_only"],
        default="r1_zero",
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default="cs336-a5-grpo")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--n-train-examples", type=int, default=6400)
    parser.add_argument("--n-val-examples", type=int, default=1024)
    parser.add_argument("--num-rollout-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--num-gradient-steps", type=int, default=1)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-max-tokens", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--vllm-gpu", type=int, default=1)
    parser.add_argument("--baseline", choices=["mean", "none"], default="mean")
    parser.add_argument(
        "--advantage-normalizer",
        choices=["std", "none", "mean"],
        default="std",
    )
    parser.add_argument(
        "--loss-normalization",
        choices=["sequence", "constant"],
        default="sequence",
    )
    parser.add_argument("--normalization-constant", type=int, default=None)
    parser.add_argument(
        "--importance-reweighting-method",
        choices=["none", "noclip", "grpo", "gspo"],
        default="none",
    )
    parser.add_argument("--cliprange", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_grpo(
        model_name=args.model_id,
        prompt_name=args.prompt,
        training_set_path=args.train_path,
        validation_set_path=args.val_path,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        n_train_examples=args.n_train_examples,
        n_val_examples=args.n_val_examples,
        num_rollout_steps=args.num_rollout_steps,
        learning_rate=args.learning_rate,
        rollout_batch_size=args.train_batch_size,
        group_size=args.group_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_gradient_steps=args.num_gradient_steps,
        sampling_temperature=args.sampling_temperature,
        sampling_max_tokens=args.sampling_max_tokens,
        max_grad_norm=args.max_grad_norm,
        val_every=args.val_every,
        policy_device=args.policy_device,
        vllm_gpu=args.vllm_gpu,
        baseline=args.baseline,
        advantage_normalizer=args.advantage_normalizer,
        loss_normalization=args.loss_normalization,
        normalization_constant=args.normalization_constant,
        importance_reweighting_method=args.importance_reweighting_method,
        cliprange=args.cliprange,
    )


if __name__ == "__main__":
    main()
