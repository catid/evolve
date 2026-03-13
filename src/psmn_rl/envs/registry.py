from __future__ import annotations

import gymnasium as gym

from psmn_rl.config import EnvConfig
from psmn_rl.envs.minigrid import build_env_fn
from psmn_rl.envs.procgen import build_env_fn as build_procgen_env_fn


def make_reset_seeds(num_envs: int, seed: int, world_rank: int = 0) -> list[int]:
    base_seed = seed + world_rank * 10_000
    return [base_seed + index for index in range(num_envs)]


def make_vector_env(config: EnvConfig, seed: int, world_rank: int = 0) -> gym.vector.VectorEnv:
    base_seed = seed + world_rank * 10_000
    if config.suite == "minigrid":
        env_fns = [
            build_env_fn(
                env_id=config.env_id,
                seed=base_seed,
                env_index=index,
                max_episode_steps=config.max_episode_steps,
                fully_observed=config.fully_observed,
            )
            for index in range(config.num_envs)
        ]
    elif config.suite == "procgen":
        env_fns = [build_procgen_env_fn(config, seed=base_seed, env_index=index) for index in range(config.num_envs)]
    else:
        raise ValueError(f"Unsupported environment suite: {config.suite}")
    return gym.vector.SyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)


def make_eval_env(config: EnvConfig, seed: int, world_rank: int = 0) -> gym.vector.VectorEnv:
    eval_cfg = EnvConfig(
        suite=config.suite,
        env_id=config.env_id,
        num_envs=config.num_eval_envs,
        num_eval_envs=config.num_eval_envs,
        max_episode_steps=config.max_episode_steps,
        capture_video=config.capture_video,
        fully_observed=config.fully_observed,
        distribution_mode=config.distribution_mode,
        start_level=config.start_level,
        num_levels=config.num_levels,
    )
    return make_vector_env(eval_cfg, seed=seed, world_rank=world_rank)
