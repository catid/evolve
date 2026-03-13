from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import torch


def get_git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
    except Exception:
        return "unknown"


def get_git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return bool(result.stdout.strip())


def try_get_gpu_utilization() -> float | None:
    if not torch.cuda.is_available():
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        if not values:
            return None
        return sum(values) / len(values)
    except Exception:
        return None


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))
