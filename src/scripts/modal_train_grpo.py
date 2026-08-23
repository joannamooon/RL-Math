import shlex

from modal_utils import app, submit_commands

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
