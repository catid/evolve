from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym

from psmn_rl.config import EnvConfig


def ensure_procgen_available() -> None:
    try:
        import procgen_gym  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Procgen support requires a Gymnasium-compatible Procgen package. "
            "On Python 3.12, the original openai/procgen package is not compatible, "
            "and the current procgen-gym port targets Python 3.13+. "
            "Run ./scripts/install_procgen_port.sh to install the pinned local port "
            "before using suite=procgen."
        ) from exc


def build_env_fn(config: EnvConfig, seed: int, env_index: int) -> Callable[[], gym.Env]:
    def thunk() -> gym.Env:
        ensure_procgen_available()
        env = gym.make(
            config.env_id,
            render_mode=None,
            distribution_mode=config.distribution_mode,
            start_level=config.start_level + env_index,
            num_levels=config.num_levels,
        )
        if config.max_episode_steps is not None:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_episode_steps)
        env.reset(seed=seed + env_index)
        return env

    return thunk
