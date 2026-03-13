from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from psmn_rl.config import load_config
from psmn_rl.train import run_training


def test_policy_diagnostics_analysis_smoke(tmp_path: Path) -> None:
    config = load_config("configs/diagnostic/minigrid_empty5_flat_dense_overfit.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 1
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 1
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "run")
    run_training(config, max_updates=1)

    report_path = tmp_path / "policy_report.md"
    csv_path = tmp_path / "policy_report.csv"
    trace_dir = tmp_path / "traces"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "psmn_rl.analysis.policy_diagnostics",
            str(config.logging.output_dir),
            "--episodes",
            "2",
            "--device",
            "cpu",
            "--output",
            str(report_path),
            "--csv",
            str(csv_path),
            "--trace-dir",
            str(trace_dir),
            "--trace-limit",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )

    assert report_path.exists()
    assert csv_path.exists()
    assert any(trace_dir.glob("*.json"))
