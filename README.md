# Alignment — Reasoning RL

Post-training a language model with GRPO (Group Relative Policy Optimization) to improve
math-reasoning accuracy on GSM8K, starting from the OLMo-2-0425-1B base model.

Built for Stanford's CS336 (Language Models from Scratch), Assignment 5.

## What's here

- **Prompting**: zero-shot / few-shot / chain-of-thought prompting of the base model on GSM8K.
- **GRPO**: on-policy policy-gradient training loop (rollout via vLLM, gradient steps via
  HuggingFace `transformers`), including group-mean baselines, advantage normalization
  (std/mean/none), and sequence vs. constant loss normalization.
- **RL algorithm variants**: Dr. GRPO, RFT (rejection fine-tuning), MaxRL — same training loop,
  different baseline/normalization choices.
- **Off-policy RL**: multiple gradient steps per rollout batch, with PPO/GRPO-style clipped
  token-level importance reweighting and GSPO-style clipped sequence-level reweighting.

## Repo layout

```
cs336_alignment/
  vllm_utils.py         vLLM server lifecycle, generation, NCCL weight sync
  checkpoint.py          load/save HF model + tokenizer
  drgrpo_grader.py        math-answer grading (r1_zero_reward_fn, question_only_reward_fn)
  prompts/                r1_zero, r1_zero_three_shot, question_only prompt templates
  prompts_safety/         prompt templates for the optional safety/RLHF supplement
tests/
  adapters.py             implementation hooks — connects your code to the test suite
  test_grpo.py             required tests
  test_data.py, test_dpo.py, test_metrics.py    optional supplement tests
scripts/
  prompting_baselines.py  eval OLMo-2-0425-1B on GSM8K across prompting strategies
  train_grpo.py            full GRPO training loop (on- and off-policy, all variants)
  evaluate_safety.py      optional supplement: LLaMA-judged safety eval
data/                      GSM8K, MMLU, AlpacaEval, HH, SimpleSafetyTests
```

## Setup

Uses `uv` for dependency management.

```sh
uv sync --no-install-package flash-attn
uv sync
```

## Running things

Prompting baselines (zero-shot `question_only`, zero-shot `r1_zero`, few-shot `r1_zero_three_shot`):

```sh
uv run python scripts/prompting_baselines.py --model-id allenai/OLMo-2-0425-1B
```

On-policy GRPO training on GSM8K:

```sh
uv run python scripts/train_grpo.py \
    --model-id allenai/OLMo-2-0425-1B \
    --prompt r1_zero \
    --wandb-project cs336-a5-grpo
```

RL variants and off-policy training reuse the same script — see the flags in
`scripts/train_grpo.py` (`--baseline`, `--advantage-normalizer`, `--loss-normalization`,
`--importance-reweighting-method`, `--cliprange`, `--train-batch-size`,
`--gradient-accumulation-steps`) and the "Empirical problems" section of `WRITEUP.md` for the
exact configs used for each experiment (Dr. GRPO, RFT, MaxRL, off-policy naive/noclip/grpo/gspo).

Training requires 2 GPUs (one for the HuggingFace policy/optimizer, one for the vLLM rollout
server) and, for the full experiment sweeps in the handout, substantial B200 GPU time.

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
| Off-policy naive (32x, no reweighting) | 18% ± 6% | biased, degrades vs. on-policy |
| Off-policy, GRPO-clip | 25% ± 4% | recovers most of on-policy performance |
| Off-policy, GSPO-clip | 26% ± 3% | most stable off-policy variant, lowest clip fraction |
