from __future__ import annotations

from pathlib import Path

import torch

from psmn_rl.config import load_config
from psmn_rl.envs.registry import make_vector_env
from psmn_rl.models.factory import build_model
from psmn_rl.rl.distributed.ddp import DistributedContext
from psmn_rl.rl.ppo.algorithm import evaluate_policy
from psmn_rl.train import run_training


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


def test_greedy_eval_reproduces_for_same_checkpoint_and_seed(tmp_path: Path) -> None:
    config = load_config("configs/diagnostic/minigrid_empty5_flat_dense_overfit.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 1
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 8
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 1
    config.evaluation.episodes = 2
    config.logging.output_dir = str(tmp_path / "train")

    run_training(config, max_updates=1)
    checkpoint_path = Path(config.logging.output_dir) / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    envs = make_vector_env(config.env, seed=config.seed)
    model = build_model(config.model, envs.single_observation_space, envs.single_action_space)
    model.load_state_dict(checkpoint["model"])
    model.train()

    ctx = _cpu_ctx()
    first = evaluate_policy(config, model, ctx, episodes=2)
    second = evaluate_policy(config, model, ctx, episodes=2)

    assert first == second
    assert model.training
    envs.close()


def test_sampled_temperature_eval_reproduces_for_same_checkpoint_and_seed(tmp_path: Path) -> None:
    config = load_config("configs/diagnostic/minigrid_empty5_token_dense_overfit.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 1
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 8
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 1
    config.evaluation.episodes = 2
    config.logging.output_dir = str(tmp_path / "train_sampled")

    run_training(config, max_updates=1)
    checkpoint_path = Path(config.logging.output_dir) / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    envs = make_vector_env(config.env, seed=config.seed)
    model = build_model(config.model, envs.single_observation_space, envs.single_action_space)
    model.load_state_dict(checkpoint["model"])

    ctx = _cpu_ctx()
    first = evaluate_policy(config, model, ctx, episodes=2, greedy=False, temperature=0.7)
    second = evaluate_policy(config, model, ctx, episodes=2, greedy=False, temperature=0.7)

    assert first == second
    assert "eval/action_max_prob" in first
    assert "eval/action_logit_margin" in first
    envs.close()
