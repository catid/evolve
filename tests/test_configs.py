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


def test_load_sanity_config() -> None:
    config = load_config(Path("configs/diagnostic/minigrid_empty5_token_dense_sanity.yaml"))
    assert config.env.env_id == "MiniGrid-Empty-5x5-v0"
    assert config.model.variant == "token_dense"


def test_load_diagnostic_memory_probe_config() -> None:
    config = load_config(Path("configs/diagnostic/minigrid_memory_token_gru_probe.yaml"))
    assert config.env.env_id == "MiniGrid-MemoryS9-v0"
    assert config.model.variant == "token_gru"
    assert config.ppo.sequence_minibatches is True


def test_load_keycorridor_recurrent_control_config() -> None:
    config = load_config(Path("configs/experiments/minigrid_keycorridor_token_gru.yaml"))
    assert config.env.env_id == "MiniGrid-KeyCorridorS3R1-v0"
    assert config.model.variant == "token_gru"
    assert config.ppo.sequence_minibatches is True


def test_load_experiment_config() -> None:
    config = load_config(Path("configs/experiments/minigrid_doorkey_sare.yaml"))
    assert config.env.env_id == "MiniGrid-DoorKey-5x5-v0"
    assert config.model.variant == "sare"


def test_load_memory_margin_decode_configs() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_margin_residual.yaml"))
    gru = load_config(Path("configs/experiments/minigrid_memory_token_gru_long_margin_residual.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_margin_residual is True
    assert gru.model.variant == "token_gru"
    assert gru.model.policy_margin_residual is True
    assert gru.ppo.sequence_minibatches is True


def test_load_memory_option_adapter_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_option_adapter.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.por_option_action_adapter is True
    assert por.model.por_option_action_adapter_scale == 0.5


def test_load_memory_hidden_residual_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_hidden_residual.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.por_option_hidden_residual is True
    assert por.model.por_option_hidden_residual_scale == 0.5


def test_load_memory_action_experts_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_action_experts.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.por_option_action_experts is True
    assert por.model.por_option_action_experts_scale == 1.0


def test_load_memory_option_film_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_option_film.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.por_option_film is True
    assert por.model.por_option_film_scale == 0.5


def test_load_memory_logit_gain_configs() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_logit_gain.yaml"))
    gru = load_config(Path("configs/experiments/minigrid_memory_token_gru_long_logit_gain.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_logit_gain is True
    assert por.model.policy_logit_gain_scale == 0.5
    assert gru.model.variant == "token_gru"
    assert gru.model.policy_logit_gain is True
    assert gru.ppo.sequence_minibatches is True


def test_load_memory_context_film_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_context_film.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.por_option_context_film is True
    assert por.model.por_option_context_film_scale == 0.5
