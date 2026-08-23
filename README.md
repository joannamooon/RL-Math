# RL-Math — Reasoning RL with GRPO

Post-training a language model with **GRPO** (Group Relative Policy Optimization) to improve math-reasoning accuracy on GSM8K, starting from the [OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B) base model.

Built for Stanford CS336 (Language Models from Scratch), Assignment 5.

## Overview

This repo implements a full RL training stack for math reasoning:

- **Rollouts** via a vLLM inference server (fast batched generation)
- **Policy updates** via HuggingFace `transformers` (gradient steps on the training GPU)
- **Weight sync** between the trainer and vLLM over NCCL

Supported training modes:

| Mode | Key flags |
|---|---|
| On-policy GRPO | `--num-gradient-steps 1` (default) |
| Dr. GRPO | `--baseline none --advantage-normalizer none --loss-normalization constant` |
| MaxRL | `--advantage-normalizer mean` |
| Off-policy (naive) | `--num-gradient-steps 32 --importance-reweighting-method none` |
| Off-policy (GRPO-clip) | `--num-gradient-steps 32 --importance-reweighting-method grpo --cliprange 0.2` |
| Off-policy (GSPO-clip) | `--num-gradient-steps 32 --importance-reweighting-method gspo --cliprange 0.2` |

## Repo layout

```
src/
  grpo.py                 GRPO utilities (tokenization, rewards, loss, train step)
  vllm_utils.py           vLLM server lifecycle, generation, NCCL weight sync
  checkpoint.py           load HF model + tokenizer
  drgrpo_grader.py        math-answer grading and reward functions
  modal_utils.py          Modal job helpers for cloud GPU runs
  prompts/                r1_zero, r1_zero_three_shot, question_only templates
  scripts/
    train_grpo.py         full GRPO training loop
    modal_train_grpo.py   launch training jobs on Modal
    prompting_baselines.py  eval base model prompting strategies on GSM8K
data/
  gsm8k/
    train.jsonl
    test.jsonl
```

## Setup

Uses [uv](https://docs.astral.sh/uv/) for dependency management. GPU extras (vLLM, flash-attn, wandb) are required for training.

```sh
uv sync --no-install-package flash-attn
uv sync --extra gpu
```

Training requires **2 GPUs**:

- `cuda:0` — HuggingFace policy + optimizer
- GPU 1 — vLLM rollout server

## Running locally

Set `PYTHONPATH=src` so scripts can import the project modules.

### Prompting baselines

Evaluate the base model with zero-shot `question_only`, zero-shot `r1_zero`, or few-shot `r1_zero_three_shot` prompts:

```sh
PYTHONPATH=src uv run python src/scripts/prompting_baselines.py
```

### On-policy GRPO training

```sh
PYTHONPATH=src uv run python src/scripts/train_grpo.py \
    --model-id allenai/OLMo-2-0425-1B \
    --prompt r1_zero \
    --seed 0 \
    --wandb-project cs336-a5-grpo
```

### RL algorithm variants

Dr. GRPO:

```sh
PYTHONPATH=src uv run python src/scripts/train_grpo.py \
    --prompt r1_zero \
    --baseline none \
    --advantage-normalizer none \
    --loss-normalization constant
```

Off-policy with GRPO-style clipping:

```sh
PYTHONPATH=src uv run python src/scripts/train_grpo.py \
    --prompt r1_zero \
    --num-gradient-steps 32 \
    --importance-reweighting-method grpo \
    --cliprange 0.2
```

Run `PYTHONPATH=src uv run python src/scripts/train_grpo.py --help` for all options.

### Training loop

Each rollout step:

1. Sample a batch of GSM8K questions and format them with the chosen prompt template
2. Sync policy weights to vLLM
3. Generate `group_size` completions per prompt (default 8)
4. Score completions with the prompt's reward function
5. Compute group-normalized advantages and run a policy-gradient update
6. Log train/val metrics to Weights & Biases

Default hyperparameters: 200 rollout steps, batch size 256 (32 prompts × 8 completions), learning rate `1e-5`, 6400 training examples.

For off-policy training, `old_log_probs` are computed once after each rollout and reused across all gradient steps in that batch, so importance ratios stay anchored to the rollout policy.

## Modal (cloud GPUs)

To launch sweeps on Modal B200 instances, set your SUNET ID in `src/modal_utils.py`, then:

```sh
uv run modal run src/scripts/modal_train_grpo.py \
    --seeds 0,1,2,3 \
    --prompt r1_zero \
    --extra-args "--baseline none --advantage-normalizer none --loss-normalization constant"
```

This submits one job per seed. W&B credentials are read from a Modal secret named `wandb`.

## Key CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model-id` | `allenai/OLMo-2-0425-1B` | Base model to train |
| `--prompt` | `r1_zero` | Prompt template (`r1_zero`, `r1_zero_three_shot`, `question_only`) |
| `--train-batch-size` | `256` | Total completions per rollout step |
| `--group-size` | `8` | Completions sampled per prompt |
| `--num-rollout-steps` | `200` | Number of rollout/update cycles |
| `--num-gradient-steps` | `1` | Gradient steps per rollout (off-policy when > 1) |
| `--gradient-accumulation-steps` | `32` | Microbatches per gradient step |
| `--baseline` | `mean` | Group baseline for advantages (`mean`, `none`) |
| `--advantage-normalizer` | `std` | Advantage scaling (`std`, `mean`, `none`) |
| `--loss-normalization` | `sequence` | Loss denominator (`sequence`, `constant`) |
| `--importance-reweighting-method` | `none` | Off-policy correction (`none`, `noclip`, `grpo`, `gspo`) |
| `--cliprange` | `None` | Clip range for `grpo` / `gspo` reweighting |
| `--policy-device` | `cuda:0` | Device for the training policy |
| `--vllm-gpu` | `1` | GPU index for the vLLM server |

## Results

| Setup | GSM8K accuracy | Notes |
|---|---|---|
| `question_only`, zero-shot | ~2% | mostly unformatted continuations, few parseable answers |
| `r1_zero`, zero-shot | ~6% | format mostly followed, reasoning shallow |
| `r1_zero_three_shot`, few-shot | ~13% | in-context examples improve both format and answers |
| GRPO (on-policy, `r1_zero`), 4 seeds | 27% ± 3% | 200 rollout steps, `lr=1e-5` |
| Dr. GRPO | 26% ± 4% | similar mean, slightly higher seed variance than GRPO |
| RFT | 22% ± 5% | slower/noisier, no negative-sample signal |
| MaxRL | 28% ± 3% | modest gain from upweighting hard prompts |
| Off-policy naive (32×, no reweighting) | 18% ± 6% | biased, degrades vs. on-policy |
| Off-policy, GRPO-clip | 25% ± 4% | recovers most of on-policy performance |
| Off-policy, GSPO-clip | 26% ± 3% | most stable off-policy variant, lowest clip fraction |
