"""
Evaluate OLMo-2-0425-1B on GSM8K with zero-shot question_only, zero-shot
r1_zero, and few-shot r1_zero_three_shot prompts (Problem: prompting_baselines).

Running:

```
uv run python scripts/prompting_baselines.py \
    --model-id allenai/OLMo-2-0425-1B \
    --output-dir results/prompting_baselines
```
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
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


def load_gsm8k(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            example = json.loads(line)
            # Ground truth answers in GSM8K are "{rationale}\n#### {answer}".
            ground_truth = example["answer"].split("####")[-1].strip()
            examples.append({"question": example["question"], "ground_truth": ground_truth})
    return examples


def categorize(reward: dict[str, float]) -> str:
    if reward["format_reward"] == 1.0 and reward["answer_reward"] == 1.0:
        return "correct_and_formatted"
    if reward["format_reward"] == 1.0 and reward["answer_reward"] == 0.0:
        return "formatted_but_wrong"
    return "unformatted"


def run_eval(server: VLLMServer, prompt_name: str, examples: list[dict], output_dir: Path) -> dict:
    config = PROMPT_CONFIGS[prompt_name]
    template = config["prompt_path"].read_text()
    prompts = [template.format(question=example["question"]) for example in examples]

    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "n": 1,
        "seed": 0,
    }
    if config["stop"] is not None:
        sampling_params["stop"] = [config["stop"]]
        sampling_params["include_stop_str_in_output"] = True

    completions = server.generate_completions(prompts=prompts, sampling_params=sampling_params, batch_size=256)

    counts = {"correct_and_formatted": 0, "formatted_but_wrong": 0, "unformatted": 0}
    records = []
    reward_fn = config["reward_fn"]
    for prompt, completion, example in zip(prompts, completions, examples):
        reward = reward_fn(completion.text, example["ground_truth"])
        category = categorize(reward)
        counts[category] += 1
        records.append(
            {
                "prompt": prompt,
                "response": completion.text,
                "ground_truth": example["ground_truth"],
                "reward": reward,
                "category": category,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{prompt_name}.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    n = len(examples)
    summary = {
        "prompt_name": prompt_name,
        "n": n,
        "accuracy": counts["correct_and_formatted"] / n,
        "format_rate": (counts["correct_and_formatted"] + counts["formatted_but_wrong"]) / n,
        **counts,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--test-path", default="data/gsm8k/test.jsonl")
    parser.add_argument("--output-dir", default="results/prompting_baselines")
    parser.add_argument("--limit", type=int, default=None, help="Optionally limit number of eval examples.")
    args = parser.parse_args()

    examples = load_gsm8k(Path(args.test_path))
    if args.limit is not None:
        examples = examples[: args.limit]

    server = VLLMServer(model_id=args.model_id, gpu=0, seed=0)
    server.start()

    output_dir = Path(args.output_dir)
    summaries = {}
    for prompt_name in PROMPT_CONFIGS:
        summary = run_eval(server, prompt_name, examples, output_dir)
        summaries[prompt_name] = summary
        print(json.dumps(summary, indent=2))

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    server.stop()


if __name__ == "__main__":
    main()
