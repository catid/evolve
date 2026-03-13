from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from psmn_rl.config import load_config
from psmn_rl.rl.distributed.ddp import DistributedContext
from psmn_rl.rl.ppo.algorithm import _apply_truncation_bootstrap
from psmn_rl.train import run_training


class _DummyValueModel(nn.Module):
    def forward(self, obs, state=None, done=None):
        pixels = obs["pixels"].float()
        value = pixels.flatten(1).sum(dim=1)
        return SimpleNamespace(value=value)


def test_truncation_bootstrap_uses_final_observation_value() -> None:
    model = _DummyValueModel()
    reward_t = torch.zeros(2, dtype=torch.float32)
    ctx = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        is_distributed=False,
        is_main_process=True,
        autocast_dtype=None,
    )
    final_obs = np.stack(
        [
            np.ones((2, 2, 1), dtype=np.float32),
            np.zeros((2, 2, 1), dtype=np.float32),
        ],
        axis=0,
    )
    _apply_truncation_bootstrap(
        model=model,
        next_state={},
        truncated=np.array([True, False]),
        info={"final_obs": final_obs, "_final_obs": np.array([True, False])},
        reward_t=reward_t,
        ctx=ctx,
        gamma=0.9,
    )
    assert reward_t[0].item() == pytest.approx(3.6)
    assert reward_t[1].item() == 0.0


def test_max_updates_override_is_written_to_resolved_config(tmp_path: Path) -> None:
    config = load_config("configs/baseline/minigrid_dense.yaml")
    config.system.device = "cpu"
    config.env.num_envs = 1
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 5
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 1
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "dense_override")

    run_training(config, max_updates=1)

    resolved = load_config(Path(config.logging.output_dir) / "resolved_config.yaml")
    assert resolved.ppo.total_updates == 1
