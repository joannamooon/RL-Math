"""
GRPO training loop for OLMo-2-0425-1B on GSM8K (Problems: grpo_experiments_standard_on_policy,
grpo_learning_rate, grpo_prompt_ablation, grpo_experiments_variants_on_policy,
grpo_experiments_off_policy, try_your_own).

Running (standard on-policy GRPO, zero-shot r1_zero prompt):

```
uv run python scripts/train_grpo.py \
    --model-id allenai/OLMo-2-0425-1B \
    --prompt r1_zero \
    --train-path data/gsm8k/train.jsonl \
    --val-path data/gsm8k/test.jsonl \
    --seed 0 \
    --wandb-project cs336-a5-grpo
```

Off-policy / variant runs are controlled via --baseline, --advantage-normalizer,
--loss-normalization, --importance-reweighting-method, --cliprange,
--train-batch-size, and --gradient-accumulation-steps.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.grpo import get_response_log_probs, grpo_train_step, tokenize_prompt_and_output
from cs336_alignment.vllm_utils import VLLMServer

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "cs336_alignment" / "prompts"

PROMPT_CONFIGS = {
    "question_only": {
        "prompt_path": PROMPTS_DIR / "question_only.prompt",
        "reward_fn": question_only_reward_fn,
        "stop": None,
    },
    "r1_zero": {
        "prompt_path": PROMPTS_DIR / "r1_zero.prompt",
        "reward_fn": r1_zero_reward_fn,
        "stop": "</answer>",
    },
    "r1_zero_three_shot": {
        "prompt_path": PROMPTS_DIR / "r1_zero_three_shot_gsm8k.prompt",
        "reward_fn": r1_zero_reward_fn,
        "stop": "</answer>",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_gsm8k(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            example = json.loads(line)
            ground_truth = example["answer"].split("####")[-1].strip()
            examples.append({"question": example["question"], "ground_truth": ground_truth})
    return examples


def make_sampling_params(prompt_name: str, max_tokens: int, temperature: float, seed: int) -> dict:
    config = PROMPT_CONFIGS[prompt_name]
    sampling_params = {
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "n": 1,
        "seed": seed,
    }
    if config["stop"] is not None:
        sampling_params["stop"] = [config["stop"]]
        sampling_params["include_stop_str_in_output"] = True
    return sampling_params


@torch.no_grad()
def evaluate(server, template, reward_fn, examples, max_tokens, sampling_seed, log_n_examples=0):
    prompts = [template.format(question=example["question"]) for example in examples]
    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "n": 1,
        "seed": sampling_seed,
    }
    completions = server.generate_completions(prompts=prompts, sampling_params=sampling_params, batch_size=256)

    rewards, format_rewards, response_lens = [], [], []
    logged = []
    for prompt, completion, example in zip(prompts, completions, examples):
        reward = reward_fn(completion.text, example["ground_truth"])
        rewards.append(reward["reward"])
        format_rewards.append(reward["format_reward"])
        response_lens.append(len(completion.token_ids))
        if len(logged) < log_n_examples:
            logged.append({"prompt": prompt, "response": completion.text, "reward": reward})

    return {
        "val/reward": float(np.mean(rewards)),
        "val/format_reward": float(np.mean(format_rewards)),
        "val/avg_response_length": float(np.mean(response_lens)),
    }, logged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--prompt", choices=list(PROMPT_CONFIGS), default="r1_zero")
    parser.add_argument("--train-path", default="data/gsm8k/train.jsonl")
    parser.add_argument("--val-path", default="data/gsm8k/test.jsonl")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--n-train-examples", type=int, default=6400)
    parser.add_argument("--n-val-examples", type=int, default=1024)
    parser.add_argument("--num-rollout-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-max-tokens", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--baseline", choices=["mean", "none"], default="mean")
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--advantage-normalizer", choices=["std", "none", "mean"], default="std")
    parser.add_argument("--loss-normalization", choices=["sequence", "constant"], default="sequence")
    parser.add_argument("--normalization-constant", type=int, default=None)

    parser.add_argument(
        "--importance-reweighting-method", choices=["none", "noclip", "grpo", "gspo"], default="none"
    )
    parser.add_argument("--cliprange", type=float, default=None)

    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--log-rollouts-every", type=int, default=40)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--output-dir", default="results/grpo")
    args = parser.parse_args()

    set_seed(args.seed)

    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_config = PROMPT_CONFIGS[args.prompt]
    template = prompt_config["prompt_path"].read_text()
    reward_fn = prompt_config["reward_fn"]

    train_examples = load_gsm8k(Path(args.train_path))[: args.n_train_examples]
    val_examples = load_gsm8k(Path(args.val_path))[: args.n_val_examples]

    model, tokenizer = get_model_and_tokenizer(args.model_id, device="cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)

    server = VLLMServer(model_id=args.model_id, gpu=1, seed=args.seed)
    server.start()
    server.init_weight_sync(policy_device="cuda:0")

    n_prompts_per_rollout_batch = args.rollout_batch_size // args.group_size
    off_policy_train_steps = args.rollout_batch_size // args.train_batch_size

    rng = random.Random(args.seed)
    train_pool = list(train_examples)

    for rollout_step in range(args.num_rollout_steps):
        server.sync_policy_weights(model)

        batch_examples = [rng.choice(train_pool) for _ in range(n_prompts_per_rollout_batch)]
        repeated_prompts_text = []
        repeated_ground_truths = []
        for example in batch_examples:
            prompt_text = template.format(question=example["question"])
            for _ in range(args.group_size):
                repeated_prompts_text.append(prompt_text)
                repeated_ground_truths.append(example["ground_truth"])

        sampling_params = make_sampling_params(
            args.prompt, args.sampling_max_tokens, args.sampling_temperature, seed=rollout_step
        )
        completions = server.generate_completions(
            prompts=repeated_prompts_text,
            sampling_params=sampling_params,
            batch_size=256,
        )
        rollout_responses = [c.text for c in completions]

        old_log_probs = None
        if args.importance_reweighting_method != "none":
            model.eval()
            with torch.no_grad():
                tokenized = tokenize_prompt_and_output(repeated_prompts_text, rollout_responses, tokenizer)
                old_log_probs = get_response_log_probs(
                    model=model,
                    input_ids=tokenized["input_ids"].to("cuda:0"),
                    labels=tokenized["labels"].to("cuda:0"),
                    return_token_entropy=False,
                )["log_probs"].cpu()
            model.train()

        model.train()
        n_micro_steps = off_policy_train_steps if args.importance_reweighting_method != "none" else 1
        for micro_step in range(n_micro_steps):
            start = micro_step * args.train_batch_size
            end = start + args.train_batch_size
            step_old_log_probs = old_log_probs[start:end] if old_log_probs is not None else None

            loss, metadata = grpo_train_step(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
                reward_fn=reward_fn,
                repeated_prompts=repeated_prompts_text[start:end],
                rollout_responses=rollout_responses[start:end],
                repeated_ground_truths=repeated_ground_truths[start:end],
                group_size=args.group_size,
                baseline=args.baseline,
                advantage_eps=args.advantage_eps,
                advantage_normalizer=args.advantage_normalizer,
                importance_reweighting_method=args.importance_reweighting_method,
                old_log_probs=step_old_log_probs,
                cliprange=args.cliprange,
                loss_normalization=args.loss_normalization,
                normalization_constant=args.normalization_constant,
            )

            log_dict = {
                "train/loss": loss.item(),
                "train/grad_norm": float(metadata["grad_norm"]),
                "train/token_entropy": float(metadata["token_entropy"]),
                "train/mean_reward": float(metadata["mean_reward"]),
                "train/mean_format_reward": float(metadata["mean_format_reward"]),
                "rollout_step": rollout_step,
                "micro_step": micro_step,
            }
            print(json.dumps(log_dict))
            if use_wandb:
                wandb.log(log_dict)

        if rollout_step % args.log_rollouts_every == 0:
            sample_records = [
                {"prompt": p, "response": r}
                for p, r in list(zip(repeated_prompts_text, rollout_responses))[:8]
            ]
            with open(output_dir / f"rollouts_step{rollout_step}.json", "w") as f:
                json.dump(sample_records, f, indent=2)

        if rollout_step % args.eval_every == 0 or rollout_step == args.num_rollout_steps - 1:
            model.eval()
            server.sync_policy_weights(model)
            val_metrics, val_examples_logged = evaluate(
                server, template, reward_fn, val_examples, args.sampling_max_tokens, sampling_seed=0, log_n_examples=4
            )
            print(json.dumps({**val_metrics, "rollout_step": rollout_step}))
            if use_wandb:
                wandb.log({**val_metrics, "rollout_step": rollout_step})
            model.train()

    final_dir = output_dir / "final_model"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    server.stop()

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
