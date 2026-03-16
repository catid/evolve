from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from psmn_rl.config import load_config
from psmn_rl.train import run_training


def test_checkpoint_archive_and_dynamics_report_smoke(tmp_path: Path) -> None:
    config = load_config("configs/experiments/minigrid_doorkey_single_expert_ent1e3_ckptscan.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 1
    config.env.num_eval_envs = 1
    config.system.checkpoint_interval = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 2
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 1
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "run")

    run_training(config, max_updates=2)

    assert Path(config.logging.output_dir, "latest.pt").exists()
    assert Path(config.logging.output_dir, "checkpoint_update_0001.pt").exists()
    assert Path(config.logging.output_dir, "checkpoint_update_0002.pt").exists()

    report_path = tmp_path / "checkpoint_report.md"
    csv_path = tmp_path / "checkpoint_report.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "psmn_rl.analysis.checkpoint_dynamics",
            str(config.logging.output_dir),
            "--episodes",
            "2",
            "--device",
            "cpu",
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
