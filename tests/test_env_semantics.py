import numpy as np

from psmn_rl.config import EnvConfig
from psmn_rl.envs.registry import make_reset_seeds, make_vector_env


def test_make_reset_seeds_do_not_overlap_across_ranks() -> None:
    rank0 = set(make_reset_seeds(num_envs=4, seed=7, world_rank=0))
    rank1 = set(make_reset_seeds(num_envs=4, seed=7, world_rank=1))
    assert rank0.isdisjoint(rank1)


def test_vector_env_uses_same_step_autoreset_on_truncation() -> None:
    config = EnvConfig(
        suite="minigrid",
        env_id="MiniGrid-Empty-5x5-v0",
        num_envs=1,
        num_eval_envs=1,
        max_episode_steps=2,
    )
    env = make_vector_env(config, seed=7)
    try:
        obs, _ = env.reset(seed=make_reset_seeds(1, 7))
        assert "image" in obs
        _, _, terminated, truncated, info = env.step(np.array([0], dtype=np.int64))
        assert not terminated[0]
        assert not truncated[0]

        next_obs, _, terminated, truncated, info = env.step(np.array([0], dtype=np.int64))
        assert not terminated[0]
        assert truncated[0]
        assert "image" in next_obs
        assert "_final_obs" in info
        assert bool(info["_final_obs"][0])
        assert "final_obs" in info
    finally:
        env.close()
