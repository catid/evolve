from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from psmn_rl.config import load_config
from psmn_rl.train import run_training


def test_self_imitation_run_and_report_smoke(tmp_path: Path) -> None:
    config = load_config("configs/diagnostic/minigrid_empty5_flat_dense_overfit.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 1
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 8
    config.ppo.total_updates = 5
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 1
    config.evaluation.episodes = 2
    config.logging.output_dir = str(tmp_path / "train")
    run_training(config, max_updates=5)

    run_dir = tmp_path / "self_imitation_run"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "psmn_rl.analysis.self_imitation",
            "run",
            "--checkpoint",
            str(Path(config.logging.output_dir) / "latest.pt"),
            "--config",
            str(Path(config.logging.output_dir) / "resolved_config.yaml"),
            "--output-dir",
            str(run_dir),
            "--device",
            "cpu",
            "--success-target",
            "1",
            "--max-episodes",
            "8",
            "--target",
            "policy_head",
            "--weighting",
            "uniform",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--eval-episodes",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["harvest_successes"] >= 1
    assert (run_dir / "latest.pt").exists()

    report_path = tmp_path / "self_imitation_report.md"
    csv_path = tmp_path / "self_imitation_report.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "psmn_rl.analysis.self_imitation",
            "report",
            str(run_dir),
            "--output",
            str(report_path),
            "--csv",
            str(csv_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )

    assert report_path.exists()
    assert csv_path.exists()
