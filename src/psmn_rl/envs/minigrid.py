from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper


def build_env_fn(
    env_id: str,
    seed: int,
    env_index: int,
    max_episode_steps: int | None = None,
    fully_observed: bool = False,
) -> Callable[[], gym.Env]:
    def thunk() -> gym.Env:
        env = gym.make(env_id, render_mode=None)
        if fully_observed:
            env = FullyObsWrapper(env)
        if max_episode_steps is not None:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env.reset(seed=seed + env_index)
        return env

    return thunk
