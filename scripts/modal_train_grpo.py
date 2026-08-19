"""
Launch scripts/train_grpo.py runs on Modal, one container per seed.

Usage:

```
uv run modal run scripts/modal_train_grpo.py --seeds 0,1,2,3
uv run modal run scripts/modal_train_grpo.py --seeds 0 --extra-args "--baseline none --advantage-normalizer none --loss-normalization constant --normalization-constant 131072"
```

Requires SUNET_ID set in cs336_alignment/modal_utils.py and a Modal `wandb` secret
(`uv run modal secret create wandb WANDB_API_KEY=<key>`).
"""

import shlex

from cs336_alignment.modal_utils import app, submit_commands

REMOTE_PYTHON = "/.uv/.venv/bin/python"


@app.local_entrypoint()
def main(
    seeds: str = "0,1,2,3",
    model_id: str = "allenai/OLMo-2-0425-1B",
    prompt: str = "r1_zero",
    wandb_project: str = "cs336-a5-grpo",
    extra_args: str = "",
) -> None:
    extra = shlex.split(extra_args)
    commands = [
        [
            REMOTE_PYTHON,
            "-u",
            "scripts/train_grpo.py",
            "--model-id",
            model_id,
            "--prompt",
            prompt,
            "--wandb-project",
            wandb_project,
            "--wandb-run-name",
            f"{prompt}-seed{seed}",
            "--seed",
            seed,
            *extra,
        ]
        for seed in seeds.split(",")
    ]
    submit_commands(commands)
