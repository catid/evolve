from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def _loopback_sockets_available() -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except PermissionError:
        return False
    try:
        sock.bind(("127.0.0.1", 0))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def test_ddp_eval_smoke(tmp_path: Path) -> None:
    if not _loopback_sockets_available():
        pytest.skip("loopback sockets are unavailable in this environment")

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
            "env_id": "MiniGrid-Empty-5x5-v0",
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
            "run_name": "ddp_eval_smoke",
            "output_dir": str(tmp_path / "run"),
            "tensorboard": False,
        },
        "evaluation": {
            "episodes": 2,
            "greedy": True,
        },
    }
    config_path = tmp_path / "ddp_eval_smoke.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    train_cmd = [
        sys.executable,
        "-m",
        "psmn_rl.train",
        "--config",
        str(config_path),
        "--device",
        "cpu",
        "--max-updates",
        "1",
    ]
    subprocess.run(
        train_cmd,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )

    checkpoint_path = tmp_path / "run" / "latest.pt"
    eval_cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "psmn_rl.evaluate",
        "--checkpoint",
        str(checkpoint_path),
        "--config",
        str(config_path),
        "--device",
        "cpu",
        "--episodes",
        "2",
        "--greedy",
        "true",
    ]
    result = subprocess.run(
        eval_cmd,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )

    combined_output = f"{result.stdout}\n{result.stderr}"
    assert combined_output.count("eval_return") == 1
