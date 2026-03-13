from pathlib import Path

import pytest

from psmn_rl.config import load_config
from psmn_rl.train import run_training


def test_procgen_coinrun_smoke(tmp_path: Path) -> None:
    pytest.importorskip("procgen_gym", reason="procgen port not installed")
    config = load_config("configs/baseline/procgen_coinrun_dense.yaml")
    config.system.device = "cpu"
    config.env.num_envs = 2
    config.env.num_eval_envs = 1
    config.env.num_levels = 1
    config.ppo.rollout_steps = 8
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 1
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "procgen_dense")

    metrics = run_training(config, max_updates=1)
    assert "eval_return" in metrics
    assert Path(config.logging.output_dir, "latest.pt").exists()
