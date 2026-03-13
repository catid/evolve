import subprocess
import sys
from pathlib import Path

import yaml


def test_ddp_smoke(tmp_path: Path) -> None:
    config = {
        "seed": 7,
        "system": {
            "device": "cpu",
            "precision": "fp32",
            "deterministic": True,
            "log_interval": 1,
            "checkpoint_interval": 1,
        },
        "env": {
            "suite": "minigrid",
            "env_id": "MiniGrid-DoorKey-5x5-v0",
            "num_envs": 1,
            "num_eval_envs": 1,
        },
        "ppo": {
            "rollout_steps": 4,
            "total_updates": 1,
            "update_epochs": 1,
            "minibatches": 1,
            "learning_rate": 0.00025,
        },
        "model": {
            "variant": "token_dense",
            "hidden_size": 64,
            "token_dim": 64,
            "encoder_layers": 1,
            "num_heads": 4,
        },
        "logging": {
            "run_name": "ddp_smoke",
            "output_dir": str(tmp_path / "ddp"),
        },
        "evaluation": {
            "episodes": 1,
            "greedy": True,
        },
    }
    config_path = tmp_path / "ddp_smoke.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "psmn_rl.train",
            "--config",
            str(config_path),
            "--device",
            "cpu",
            "--max-updates",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "update=1" in result.stdout or "update=1" in result.stderr
