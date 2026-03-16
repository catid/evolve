from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from psmn_rl.config import load_config
from psmn_rl.train import run_training


def test_learner_state_supervision_run_and_report_smoke(tmp_path: Path) -> None:
    teacher_config = load_config("configs/diagnostic/minigrid_empty5_flat_dense_overfit.yaml")
    teacher_config.system.device = "cpu"
    teacher_config.logging.tensorboard = False
    teacher_config.env.num_envs = 1
    teacher_config.env.num_eval_envs = 1
    teacher_config.ppo.rollout_steps = 8
    teacher_config.ppo.total_updates = 4
    teacher_config.ppo.update_epochs = 1
    teacher_config.ppo.minibatches = 1
    teacher_config.evaluation.episodes = 2
    teacher_config.logging.output_dir = str(tmp_path / "teacher")
    run_training(teacher_config, max_updates=4)

    student_config = load_config("configs/diagnostic/minigrid_empty5_token_dense_overfit.yaml")
    student_config.system.device = "cpu"
    student_config.logging.tensorboard = False
    student_config.env.num_envs = 1
    student_config.env.num_eval_envs = 1
    student_config.ppo.rollout_steps = 8
    student_config.ppo.total_updates = 2
    student_config.ppo.update_epochs = 1
    student_config.ppo.minibatches = 1
    student_config.evaluation.episodes = 2
    student_config.logging.output_dir = str(tmp_path / "student")
    run_training(student_config, max_updates=2)

    run_dir = tmp_path / "learner_state_run"
    spec_path = tmp_path / "learner_state_spec.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "name": "smoke_lss_flat_to_token",
                "output_dir": str(run_dir),
                "teacher": {
                    "config": str(Path(teacher_config.logging.output_dir) / "resolved_config.yaml"),
                    "checkpoint": str(Path(teacher_config.logging.output_dir) / "latest.pt"),
                    "greedy": True,
                    "temperature": 1.0,
                },
                "student": {
                    "config": str(Path(student_config.logging.output_dir) / "resolved_config.yaml"),
                    "checkpoint": str(Path(student_config.logging.output_dir) / "latest.pt"),
                    "target": "policy_head",
                    "loss": "ce",
                    "weighting": "uniform",
                    "learning_rate": 1e-4,
                    "batch_size": 8,
                    "epochs": 1,
                },
                "loop": {
                    "rounds": 1,
                    "episodes_per_round": 2,
                    "max_episodes_per_round": 2,
                },
                "evaluation": {"episodes": 2},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "psmn_rl.analysis.learner_state_supervision",
            "run",
            "--spec",
            str(spec_path),
            "--device",
            "cpu",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["best_round_greedy_success"] >= 0.0
    assert summary["aggregation"] == "append_all"
    assert summary["loss"] == "ce"
    assert (run_dir / "latest.pt").exists()
    dataset = torch.load(run_dir / "round_01_dataset.pt", map_location="cpu", weights_only=False)
    assert "round_diagnostics" in dataset
    assert dataset["round_diagnostics"]["collection/steps"] > 0.0
    assert "collection/disagreement_rate" in dataset["round_diagnostics"]
    assert "teacher_confidence" in dataset

    report_path = tmp_path / "learner_state_report.md"
    csv_path = tmp_path / "learner_state_report.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "psmn_rl.analysis.learner_state_supervision",
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
