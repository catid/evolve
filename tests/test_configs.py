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


def test_load_memory_top2_rerank_configs() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_top2_rerank.yaml"))
    gru = load_config(Path("configs/experiments/minigrid_memory_token_gru_long_top2_rerank.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_top2_rerank is True
    assert por.model.policy_top2_rerank_scale == 0.5
    assert gru.model.variant == "token_gru"
    assert gru.model.policy_top2_rerank is True
    assert gru.ppo.sequence_minibatches is True


def test_load_memory_context_gain_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_context_gain.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.por_option_context_gain is True
    assert por.model.por_option_context_gain_scale == 0.5


def test_load_memory_context_film_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_context_film.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.por_option_context_film is True
    assert por.model.por_option_context_film_scale == 0.5


def test_load_memory_context_logits_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_context_logits.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.por_option_context_logits is True
    assert por.model.por_option_context_logits_scale == 0.5


def test_load_memory_option_margin_adapter_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_option_margin_adapter.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.por_option_margin_adapter is True
    assert por.model.policy_option_margin_adapter is True
    assert por.model.policy_option_margin_adapter_scale == 0.5


def test_load_memory_option_top2_rerank_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_option_top2_rerank.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.por_option_top2_rerank is True
    assert por.model.policy_option_top2_rerank is True
    assert por.model.policy_option_top2_rerank_scale == 0.5


def test_load_memory_option_top2_duration_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_option_top2_duration.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.por_option_top2_rerank is True
    assert por.model.policy_option_top2_rerank is True
    assert por.model.policy_option_top2_use_duration_gate is True


def test_load_memory_actor_hidden_film_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_film.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.por_option_actor_features is True
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_film_scale == 0.5
    assert por.model.policy_option_hidden_use_duration_gate is True


def test_load_memory_actor_hidden_film_mixed_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_film_mixed50.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.por_option_actor_features is True
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_use_duration_gate is True
    assert por.model.policy_option_hidden_duration_mix == 0.5


def test_load_memory_actor_hidden_scale_film_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_scale_film.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.por_option_actor_features is True
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_scale_only is True


def test_load_memory_actor_hidden_partial_shift_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift25.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_scale_only is False
    assert por.model.policy_option_hidden_film_scale == 0.35
    assert por.model.policy_option_hidden_shift_weight == 0.25


def test_load_memory_actor_hidden_lower_shift_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift15.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_scale_only is False
    assert por.model.policy_option_hidden_film_scale == 0.35
    assert por.model.policy_option_hidden_shift_weight == 0.15


def test_load_memory_actor_hidden_upper_shift_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift30.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_scale_only is False
    assert por.model.policy_option_hidden_film_scale == 0.35
    assert por.model.policy_option_hidden_shift_weight == 0.30


def test_load_memory_actor_hidden_mid_shift_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift225.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_scale_only is False
    assert por.model.policy_option_hidden_film_scale == 0.35
    assert por.model.policy_option_hidden_shift_weight == 0.225


def test_load_memory_actor_hidden_threshold_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_scale_only is False
    assert por.model.policy_option_hidden_film_scale == 0.35
    assert por.model.policy_option_hidden_shift_weight == 0.22


def test_load_memory_actor_hidden_centered_shift_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_centered_shift2x.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_film_scale == 0.35
    assert por.model.policy_option_hidden_shift_weight == 0.22
    assert por.model.policy_option_hidden_center_shift is True
    assert por.model.policy_option_hidden_center_shift_scale == 2.0


def test_load_memory_actor_hidden_bounded_shift_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_bounded_shift2x.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_film_scale == 0.35
    assert por.model.policy_option_hidden_shift_weight == 0.22
    assert por.model.policy_option_hidden_bound_shift is True
    assert por.model.policy_option_hidden_shift_bound_scale == 2.0


def test_load_memory_actor_hidden_split_heads_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_split_heads.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_film_scale == 0.35
    assert por.model.policy_option_hidden_use_duration_gate is True
    assert por.model.policy_option_hidden_split_heads is True
    assert por.model.policy_option_hidden_shift_weight == 0.22


def test_load_memory_actor_hidden_branch_gate_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftmix95.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_use_duration_gate is True
    assert por.model.policy_option_hidden_branch_gates is True
    assert por.model.policy_option_hidden_scale_duration_mix == 1.0
    assert por.model.policy_option_hidden_shift_duration_mix == 0.95
    assert por.model.policy_option_hidden_shift_weight == 0.22


def test_load_memory_actor_hidden_low_rank_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_lowrank16.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_use_duration_gate is True
    assert por.model.policy_option_hidden_low_rank is True
    assert por.model.policy_option_hidden_low_rank_dim == 16
    assert por.model.policy_option_hidden_shift_weight == 0.22


def test_load_memory_actor_hidden_shift22_scale_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_scale345.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_scale_only is False
    assert por.model.policy_option_hidden_film_scale == 0.345
    assert por.model.policy_option_hidden_shift_weight == 0.22


def test_load_memory_actor_hidden_shift22_scale_up_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_scale36.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_scale_only is False
    assert por.model.policy_option_hidden_film_scale == 0.36
    assert por.model.policy_option_hidden_shift_weight == 0.22


def test_load_memory_actor_hidden_micro_shift_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift221.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_scale_only is False
    assert por.model.policy_option_hidden_film_scale == 0.35
    assert por.model.policy_option_hidden_shift_weight == 0.221


def test_load_memory_actor_hidden_shift221_scale_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift221_scale40.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_scale_only is False
    assert por.model.policy_option_hidden_film_scale == 0.4
    assert por.model.policy_option_hidden_shift_weight == 0.221


def test_load_memory_actor_hidden_scale_followup_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift225_scale375.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_scale_only is False
    assert por.model.policy_option_hidden_film_scale == 0.375
    assert por.model.policy_option_hidden_shift_weight == 0.225


def test_load_memory_actor_hidden_low_scale_followup_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift225_scale325.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_scale_only is False
    assert por.model.policy_option_hidden_film_scale == 0.325
    assert por.model.policy_option_hidden_shift_weight == 0.225


def test_load_memory_actor_hidden_scale_weight_followup_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift225_scaleweight90.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_scale_only is False
    assert por.model.policy_option_hidden_film_scale == 0.35
    assert por.model.policy_option_hidden_scale_weight == 0.90
    assert por.model.policy_option_hidden_shift_weight == 0.225


def test_load_memory_actor_hidden_gate_mix_followup_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift225_mix90.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_use_duration_gate is True
    assert por.model.policy_option_hidden_film_scale == 0.35
    assert por.model.policy_option_hidden_duration_mix == 0.90
    assert por.model.policy_option_hidden_shift_weight == 0.225


def test_load_memory_actor_hidden_partial_shift_scale_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift25_scale30.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_scale_only is False
    assert por.model.policy_option_hidden_film_scale == 0.30
    assert por.model.policy_option_hidden_shift_weight == 0.25


def test_load_memory_actor_hidden_post_norm_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift25_post_norm.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_post_norm is True
    assert por.model.policy_option_hidden_film_scale == 0.35
    assert por.model.policy_option_hidden_shift_weight == 0.25


def test_load_memory_actor_hidden_split_scale_weight_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift25_scaleweight50.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_film_scale == 0.35
    assert por.model.policy_option_hidden_scale_weight == 0.5
    assert por.model.policy_option_hidden_shift_weight == 0.25


def test_load_memory_actor_hidden_low_margin_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift25_low_margin35.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_low_margin_gate is True
    assert por.model.policy_option_hidden_margin_threshold == 0.35
    assert por.model.policy_option_hidden_shift_weight == 0.25


def test_load_memory_actor_hidden_blend_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift25_blend100.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_blend_gate is True
    assert por.model.policy_option_hidden_blend_scale == 1.0
    assert por.model.policy_option_hidden_shift_weight == 0.25
