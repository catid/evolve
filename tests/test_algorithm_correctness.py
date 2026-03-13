from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from psmn_rl.config import load_config
from psmn_rl.models.factory import build_model
from psmn_rl.rl.distributed.ddp import DistributedContext
from psmn_rl.rl.ppo.algorithm import _apply_truncation_bootstrap, evaluate_policy
from psmn_rl.train import run_training


class _ConstantValueModel(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, obs, state=None, done=None):
        batch_size = next(iter(obs.values())).shape[0]
        return type("Output", (), {"value": torch.full((batch_size,), self.value, device=next(iter(obs.values())).device)})


def _cpu_ctx() -> DistributedContext:
    return DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        is_distributed=False,
        is_main_process=True,
        autocast_dtype=None,
    )


def test_truncation_bootstrap_adds_terminal_value() -> None:
    ctx = _cpu_ctx()
    reward_t = torch.zeros(3, dtype=torch.float32)
    info = {
        "final_obs": np.array([None, np.array([1.0], dtype=np.float32), None], dtype=object),
        "_final_obs": np.array([False, True, False], dtype=bool),
    }

    _apply_truncation_bootstrap(
        model=_ConstantValueModel(5.0),
        next_state={},
        truncated=np.array([False, True, False], dtype=bool),
        info=info,
        reward_t=reward_t,
        ctx=ctx,
        gamma=0.99,
    )

    assert reward_t.tolist() == pytest.approx([0.0, 4.95, 0.0])


def test_evaluate_policy_sampling_is_reproducible() -> None:
    config = load_config("configs/baseline/minigrid_dense.yaml")
    config.system.device = "cpu"
    config.env.num_eval_envs = 1
    config.evaluation.episodes = 2
    config.evaluation.greedy = False

    env = gym.make(config.env.env_id)
    model = build_model(config.model, env.observation_space, env.action_space)
    model.train()
    ctx = _cpu_ctx()

    first = evaluate_policy(config, model, ctx, episodes=2)
    second = evaluate_policy(config, model, ctx, episodes=2)

    assert first == second
    assert model.training
    env.close()


def test_resume_training_advances_global_step(tmp_path: Path) -> None:
    config = load_config("configs/baseline/minigrid_dense.yaml")
    config.system.device = "cpu"
    config.system.checkpoint_interval = 1
    config.logging.tensorboard = False
    config.env.env_id = "MiniGrid-Empty-5x5-v0"
    config.env.num_envs = 1
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 1
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "first")

    first_metrics = run_training(config, max_updates=1)
    checkpoint = Path(config.logging.output_dir) / "latest.pt"
    assert checkpoint.exists()
    assert first_metrics["global_step"] == 4.0

    resumed = load_config("configs/baseline/minigrid_dense.yaml")
    resumed.system.device = "cpu"
    resumed.system.checkpoint_interval = 1
    resumed.system.resume_from = str(checkpoint)
    resumed.logging.tensorboard = False
    resumed.env.env_id = "MiniGrid-Empty-5x5-v0"
    resumed.env.num_envs = 1
    resumed.env.num_eval_envs = 1
    resumed.ppo.rollout_steps = 4
    resumed.ppo.total_updates = 2
    resumed.ppo.update_epochs = 1
    resumed.ppo.minibatches = 1
    resumed.evaluation.episodes = 1
    resumed.logging.output_dir = str(tmp_path / "resumed")

    resumed_metrics = run_training(resumed)
    assert resumed_metrics["global_step"] == 8.0
