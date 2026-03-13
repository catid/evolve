from pathlib import Path

from psmn_rl.config import load_config


def test_load_dense_config() -> None:
    config = load_config(Path("configs/baseline/minigrid_dense.yaml"))
    assert config.model.variant == "token_dense"
    assert config.env.env_id == "MiniGrid-DoorKey-5x5-v0"
    assert config.ppo.rollout_steps == 32


def test_load_sare_config() -> None:
    config = load_config(Path("configs/sare/minigrid_doorkey.yaml"))
    assert config.model.variant == "sare"
    assert config.model.expert_count == 4
    assert "sare" in config.tags


def test_load_procgen_config() -> None:
    config = load_config(Path("configs/baseline/procgen_coinrun_dense.yaml"))
    assert config.env.suite == "procgen"
    assert config.model.patch_size == 8
    assert config.env.env_id == "procgen_gym/procgen-coinrun-v0"


def test_load_minigrid_suite_config() -> None:
    config = load_config(Path("configs/sare/minigrid_memory.yaml"))
    assert config.env.env_id == "MiniGrid-MemoryS9-v0"
    assert config.model.variant == "sare"
