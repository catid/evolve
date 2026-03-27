import math
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


def test_load_memory_actor_hidden_gate_bias_configs() -> None:
    por10 = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_gatebias10.yaml"))
    por20 = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_gatebias20.yaml"))

    assert por10.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por10.model.variant == "por"
    assert por10.model.policy_option_hidden_film is True
    assert por10.model.policy_option_hidden_gate_bias is True
    assert por10.model.policy_option_hidden_gate_bias_scale == 0.10

    assert por20.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por20.model.variant == "por"
    assert por20.model.policy_option_hidden_film is True
    assert por20.model.policy_option_hidden_gate_bias is True
    assert por20.model.policy_option_hidden_gate_bias_scale == 0.20
    assert por20.model.policy_option_hidden_shift_weight == 0.22


def test_load_memory_actor_hidden_branch_gate_bias_configs() -> None:
    scale = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_scalebias10.yaml")
    )
    shift = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftbias10.yaml")
    )

    assert scale.env.env_id == "MiniGrid-MemoryS9-v0"
    assert scale.model.variant == "por"
    assert scale.model.policy_option_hidden_film is True
    assert scale.model.policy_option_hidden_gate_bias is True
    assert scale.model.policy_option_hidden_scale_gate_bias_scale == 0.10
    assert scale.model.policy_option_hidden_shift_gate_bias_scale == 0.0

    assert shift.env.env_id == "MiniGrid-MemoryS9-v0"
    assert shift.model.variant == "por"
    assert shift.model.policy_option_hidden_film is True
    assert shift.model.policy_option_hidden_gate_bias is True
    assert shift.model.policy_option_hidden_scale_gate_bias_scale == 0.0
    assert shift.model.policy_option_hidden_shift_gate_bias_scale == 0.10


def test_load_memory_actor_hidden_adaptive_shift_floor_configs() -> None:
    floor16 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftfloor16.yaml")
    )
    floor18 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftfloor18.yaml")
    )

    assert floor16.env.env_id == "MiniGrid-MemoryS9-v0"
    assert floor16.model.variant == "por"
    assert floor16.model.policy_option_hidden_film is True
    assert floor16.model.policy_option_hidden_adaptive_shift_floor is True
    assert floor16.model.policy_option_hidden_shift_floor == 0.16

    assert floor18.env.env_id == "MiniGrid-MemoryS9-v0"
    assert floor18.model.variant == "por"
    assert floor18.model.policy_option_hidden_film is True
    assert floor18.model.policy_option_hidden_adaptive_shift_floor is True
    assert floor18.model.policy_option_hidden_shift_floor == 0.18


def test_load_memory_actor_hidden_shift_gate_power_configs() -> None:
    gate125 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate125.yaml")
    )
    gate150 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate150.yaml")
    )

    assert gate125.env.env_id == "MiniGrid-MemoryS9-v0"
    assert gate125.model.variant == "por"
    assert gate125.model.policy_option_hidden_film is True
    assert gate125.model.policy_option_hidden_shift_gate_power == 1.25
    assert gate125.model.policy_option_hidden_scale_gate_power == 1.0

    assert gate150.env.env_id == "MiniGrid-MemoryS9-v0"
    assert gate150.model.variant == "por"
    assert gate150.model.policy_option_hidden_film is True
    assert gate150.model.policy_option_hidden_shift_gate_power == 1.50
    assert gate150.model.policy_option_hidden_scale_gate_power == 1.0


def test_load_memory_actor_hidden_shift_gate_soft_configs() -> None:
    gate075 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075.yaml")
    )
    gate050 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate050.yaml")
    )

    assert gate075.env.env_id == "MiniGrid-MemoryS9-v0"
    assert gate075.model.variant == "por"
    assert gate075.model.policy_option_hidden_film is True
    assert gate075.model.policy_option_hidden_shift_gate_power == 0.75
    assert gate075.model.policy_option_hidden_scale_gate_power == 1.0

    assert gate050.env.env_id == "MiniGrid-MemoryS9-v0"
    assert gate050.model.variant == "por"
    assert gate050.model.policy_option_hidden_film is True
    assert gate050.model.policy_option_hidden_shift_gate_power == 0.50
    assert gate050.model.policy_option_hidden_scale_gate_power == 1.0


def test_load_memory_actor_hidden_shiftgate075_shift_weight_refinement_configs() -> None:
    shift215 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift215_shiftgate075.yaml")
    )
    shift225 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift225_shiftgate075.yaml")
    )

    assert shift215.env.env_id == "MiniGrid-MemoryS9-v0"
    assert shift215.model.variant == "por"
    assert shift215.model.policy_option_hidden_film is True
    assert shift215.model.policy_option_hidden_shift_weight == 0.215
    assert shift215.model.policy_option_hidden_shift_gate_power == 0.75

    assert shift225.env.env_id == "MiniGrid-MemoryS9-v0"
    assert shift225.model.variant == "por"
    assert shift225.model.policy_option_hidden_film is True
    assert shift225.model.policy_option_hidden_shift_weight == 0.225
    assert shift225.model.policy_option_hidden_shift_gate_power == 0.75


def test_load_memory_actor_hidden_shiftgate075_power_refinement_configs() -> None:
    gate070 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate070.yaml")
    )
    gate085 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate085.yaml")
    )

    assert gate070.env.env_id == "MiniGrid-MemoryS9-v0"
    assert gate070.model.variant == "por"
    assert gate070.model.policy_option_hidden_film is True
    assert gate070.model.policy_option_hidden_shift_weight == 0.22
    assert gate070.model.policy_option_hidden_shift_gate_power == 0.70

    assert gate085.env.env_id == "MiniGrid-MemoryS9-v0"
    assert gate085.model.variant == "por"
    assert gate085.model.policy_option_hidden_film is True
    assert gate085.model.policy_option_hidden_shift_weight == 0.22
    assert gate085.model.policy_option_hidden_shift_gate_power == 0.85


def test_load_memory_actor_hidden_shiftgate075_scale_refinement_configs() -> None:
    scale325 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325.yaml")
    )
    scale375 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale375.yaml")
    )

    assert scale325.env.env_id == "MiniGrid-MemoryS9-v0"
    assert scale325.model.variant == "por"
    assert scale325.model.policy_option_hidden_film is True
    assert scale325.model.policy_option_hidden_film_scale == 0.325
    assert scale325.model.policy_option_hidden_shift_gate_power == 0.75

    assert scale375.env.env_id == "MiniGrid-MemoryS9-v0"
    assert scale375.model.variant == "por"
    assert scale375.model.policy_option_hidden_film is True
    assert scale375.model.policy_option_hidden_film_scale == 0.375
    assert scale375.model.policy_option_hidden_shift_gate_power == 0.75


def test_load_memory_actor_hidden_shiftgate075_scale325_shift_refinement_configs() -> None:
    shift215 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift215_shiftgate075_scale325.yaml")
    )
    shift225 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift225_shiftgate075_scale325.yaml")
    )

    assert shift215.env.env_id == "MiniGrid-MemoryS9-v0"
    assert shift215.model.variant == "por"
    assert shift215.model.policy_option_hidden_film is True
    assert shift215.model.policy_option_hidden_film_scale == 0.325
    assert shift215.model.policy_option_hidden_shift_weight == 0.215
    assert shift215.model.policy_option_hidden_shift_gate_power == 0.75

    assert shift225.env.env_id == "MiniGrid-MemoryS9-v0"
    assert shift225.model.variant == "por"
    assert shift225.model.policy_option_hidden_film is True
    assert shift225.model.policy_option_hidden_film_scale == 0.325
    assert shift225.model.policy_option_hidden_shift_weight == 0.225
    assert shift225.model.policy_option_hidden_shift_gate_power == 0.75


def test_load_memory_actor_hidden_shiftgate075_scale325_nearby_scale_configs() -> None:
    scale3125 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale3125.yaml")
    )
    scale3375 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale3375.yaml")
    )

    assert scale3125.env.env_id == "MiniGrid-MemoryS9-v0"
    assert scale3125.model.variant == "por"
    assert scale3125.model.policy_option_hidden_film is True
    assert scale3125.model.policy_option_hidden_film_scale == 0.3125
    assert scale3125.model.policy_option_hidden_shift_weight == 0.22
    assert scale3125.model.policy_option_hidden_shift_gate_power == 0.75

    assert scale3375.env.env_id == "MiniGrid-MemoryS9-v0"
    assert scale3375.model.variant == "por"
    assert scale3375.model.policy_option_hidden_film is True
    assert scale3375.model.policy_option_hidden_film_scale == 0.3375
    assert scale3375.model.policy_option_hidden_shift_weight == 0.22
    assert scale3375.model.policy_option_hidden_shift_gate_power == 0.75


def test_load_memory_actor_hidden_shiftgate075_scale325_power_refinement_configs() -> None:
    gate0725 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate0725_scale325.yaml")
    )
    gate0775 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate0775_scale325.yaml")
    )

    assert gate0725.env.env_id == "MiniGrid-MemoryS9-v0"
    assert gate0725.model.variant == "por"
    assert gate0725.model.policy_option_hidden_film is True
    assert gate0725.model.policy_option_hidden_film_scale == 0.325
    assert gate0725.model.policy_option_hidden_shift_weight == 0.22
    assert gate0725.model.policy_option_hidden_shift_gate_power == 0.725

    assert gate0775.env.env_id == "MiniGrid-MemoryS9-v0"
    assert gate0775.model.variant == "por"
    assert gate0775.model.policy_option_hidden_film is True
    assert gate0775.model.policy_option_hidden_film_scale == 0.325
    assert gate0775.model.policy_option_hidden_shift_weight == 0.22
    assert gate0775.model.policy_option_hidden_shift_gate_power == 0.775


def test_load_memory_actor_hidden_shiftgate075_scale325_scale_gate_refinement_configs() -> None:
    scalegate090 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325_scalegate090.yaml")
    )
    scalegate110 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325_scalegate110.yaml")
    )

    assert scalegate090.env.env_id == "MiniGrid-MemoryS9-v0"
    assert scalegate090.model.variant == "por"
    assert scalegate090.model.policy_option_hidden_film is True
    assert scalegate090.model.policy_option_hidden_film_scale == 0.325
    assert scalegate090.model.policy_option_hidden_scale_gate_power == 0.90
    assert scalegate090.model.policy_option_hidden_shift_gate_power == 0.75

    assert scalegate110.env.env_id == "MiniGrid-MemoryS9-v0"
    assert scalegate110.model.variant == "por"
    assert scalegate110.model.policy_option_hidden_film is True
    assert scalegate110.model.policy_option_hidden_film_scale == 0.325
    assert scalegate110.model.policy_option_hidden_scale_gate_power == 1.10
    assert scalegate110.model.policy_option_hidden_shift_gate_power == 0.75


def test_load_memory_actor_hidden_shiftgate075_scale325_scale_gate_micro_configs() -> None:
    scalegate095 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325_scalegate095.yaml")
    )
    scalegate105 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325_scalegate105.yaml")
    )

    assert scalegate095.env.env_id == "MiniGrid-MemoryS9-v0"
    assert scalegate095.model.variant == "por"
    assert scalegate095.model.policy_option_hidden_film is True
    assert scalegate095.model.policy_option_hidden_film_scale == 0.325
    assert scalegate095.model.policy_option_hidden_scale_gate_power == 0.95
    assert scalegate095.model.policy_option_hidden_shift_gate_power == 0.75

    assert scalegate105.env.env_id == "MiniGrid-MemoryS9-v0"
    assert scalegate105.model.variant == "por"
    assert scalegate105.model.policy_option_hidden_film is True
    assert scalegate105.model.policy_option_hidden_film_scale == 0.325
    assert scalegate105.model.policy_option_hidden_scale_gate_power == 1.05
    assert scalegate105.model.policy_option_hidden_shift_gate_power == 0.75


def test_load_memory_actor_hidden_shiftgate075_scale325_branch_gate_configs() -> None:
    shiftmix95 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325_shiftmix95.yaml")
    )
    shiftmix90 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325_shiftmix90.yaml")
    )

    assert shiftmix95.env.env_id == "MiniGrid-MemoryS9-v0"
    assert shiftmix95.model.variant == "por"
    assert shiftmix95.model.policy_option_hidden_film is True
    assert shiftmix95.model.policy_option_hidden_branch_gates is True
    assert shiftmix95.model.policy_option_hidden_film_scale == 0.325
    assert shiftmix95.model.policy_option_hidden_scale_duration_mix == 1.0
    assert shiftmix95.model.policy_option_hidden_shift_duration_mix == 0.95
    assert shiftmix95.model.policy_option_hidden_shift_gate_power == 0.75

    assert shiftmix90.env.env_id == "MiniGrid-MemoryS9-v0"
    assert shiftmix90.model.variant == "por"
    assert shiftmix90.model.policy_option_hidden_film is True
    assert shiftmix90.model.policy_option_hidden_branch_gates is True
    assert shiftmix90.model.policy_option_hidden_film_scale == 0.325
    assert shiftmix90.model.policy_option_hidden_scale_duration_mix == 1.0
    assert shiftmix90.model.policy_option_hidden_shift_duration_mix == 0.90
    assert shiftmix90.model.policy_option_hidden_shift_gate_power == 0.75


def test_load_memory_actor_hidden_scale_gate_power_configs() -> None:
    gate125 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_scalegate125.yaml")
    )
    gate150 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_scalegate150.yaml")
    )

    assert gate125.env.env_id == "MiniGrid-MemoryS9-v0"
    assert gate125.model.variant == "por"
    assert gate125.model.policy_option_hidden_film is True
    assert gate125.model.policy_option_hidden_scale_gate_power == 1.25
    assert gate125.model.policy_option_hidden_shift_gate_power == 1.0

    assert gate150.env.env_id == "MiniGrid-MemoryS9-v0"
    assert gate150.model.variant == "por"
    assert gate150.model.policy_option_hidden_film is True
    assert gate150.model.policy_option_hidden_scale_gate_power == 1.50
    assert gate150.model.policy_option_hidden_shift_gate_power == 1.0


def test_load_memory_actor_hidden_scale_gate_soft_configs() -> None:
    gate075 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_scalegate075.yaml")
    )
    gate050 = load_config(
        Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_scalegate050.yaml")
    )

    assert gate075.env.env_id == "MiniGrid-MemoryS9-v0"
    assert gate075.model.variant == "por"
    assert gate075.model.policy_option_hidden_film is True
    assert gate075.model.policy_option_hidden_scale_gate_power == 0.75
    assert gate075.model.policy_option_hidden_shift_gate_power == 1.0

    assert gate050.env.env_id == "MiniGrid-MemoryS9-v0"
    assert gate050.model.variant == "por"
    assert gate050.model.policy_option_hidden_film is True
    assert gate050.model.policy_option_hidden_scale_gate_power == 0.50
    assert gate050.model.policy_option_hidden_shift_gate_power == 1.0


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


def test_load_memory_actor_hidden_termination_bias_config() -> None:
    por075 = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_termbias075.yaml"))
    por125 = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_termbias125.yaml"))

    assert por075.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por075.model.variant == "por"
    assert por075.model.policy_option_hidden_film is True
    assert por075.model.policy_option_hidden_shift_weight == 0.22
    assert por075.model.termination_bias == 0.75

    assert por125.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por125.model.variant == "por"
    assert por125.model.policy_option_hidden_film is True
    assert por125.model.policy_option_hidden_shift_weight == 0.22
    assert por125.model.termination_bias == 1.25


def test_load_memory_actor_hidden_adaptive_scale_floor_config() -> None:
    por25 = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_adaptive_floor25.yaml"))
    por30 = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_adaptive_floor30.yaml"))

    assert por25.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por25.model.variant == "por"
    assert por25.model.policy_option_hidden_film is True
    assert por25.model.policy_option_hidden_shift_weight == 0.22
    assert por25.model.policy_option_hidden_adaptive_scale_floor is True
    assert por25.model.policy_option_hidden_scale_floor == 0.25

    assert por30.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por30.model.variant == "por"
    assert por30.model.policy_option_hidden_film is True
    assert por30.model.policy_option_hidden_shift_weight == 0.22
    assert por30.model.policy_option_hidden_adaptive_scale_floor is True
    assert por30.model.policy_option_hidden_scale_floor == 0.30


def test_load_memory_actor_hidden_adaptive_shift_compensation_config() -> None:
    por50 = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_adaptive_floor25_shiftcomp50.yaml"))
    por100 = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_adaptive_floor25_shiftcomp100.yaml"))

    assert por50.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por50.model.variant == "por"
    assert por50.model.policy_option_hidden_film is True
    assert por50.model.policy_option_hidden_adaptive_scale_floor is True
    assert por50.model.policy_option_hidden_scale_floor == 0.25
    assert por50.model.policy_option_hidden_shift_compensation is True
    assert por50.model.policy_option_hidden_shift_compensation_scale == 0.5

    assert por100.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por100.model.variant == "por"
    assert por100.model.policy_option_hidden_film is True
    assert por100.model.policy_option_hidden_adaptive_scale_floor is True
    assert por100.model.policy_option_hidden_scale_floor == 0.25
    assert por100.model.policy_option_hidden_shift_compensation is True
    assert por100.model.policy_option_hidden_shift_compensation_scale == 1.0


def test_load_memory_actor_hidden_powered_adaptive_scale_floor_config() -> None:
    por075 = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_adaptive_floor25_power075.yaml"))
    por05 = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_adaptive_floor25_power05.yaml"))

    assert por075.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por075.model.variant == "por"
    assert por075.model.policy_option_hidden_adaptive_scale_floor is True
    assert por075.model.policy_option_hidden_scale_floor == 0.25
    assert por075.model.policy_option_hidden_scale_floor_power == 0.75

    assert por05.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por05.model.variant == "por"
    assert por05.model.policy_option_hidden_adaptive_scale_floor is True
    assert por05.model.policy_option_hidden_scale_floor == 0.25
    assert por05.model.policy_option_hidden_scale_floor_power == 0.5


def test_load_memory_actor_hidden_adaptive_shift_compensation_calibration_config() -> None:
    por20 = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_adaptive_floor25_shiftcomp20.yaml"))
    por35 = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_adaptive_floor25_shiftcomp35.yaml"))

    assert por20.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por20.model.variant == "por"
    assert por20.model.policy_option_hidden_adaptive_scale_floor is True
    assert por20.model.policy_option_hidden_scale_floor == 0.25
    assert por20.model.policy_option_hidden_shift_compensation is True
    assert por20.model.policy_option_hidden_shift_compensation_scale == 0.2

    assert por35.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por35.model.variant == "por"
    assert por35.model.policy_option_hidden_adaptive_scale_floor is True
    assert por35.model.policy_option_hidden_scale_floor == 0.25
    assert por35.model.policy_option_hidden_shift_compensation is True
    assert por35.model.policy_option_hidden_shift_compensation_scale == 0.35


def test_load_memory_actor_hidden_option_count_config() -> None:
    por6 = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_option6.yaml"))
    por8 = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_option8.yaml"))

    assert por6.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por6.model.variant == "por"
    assert por6.model.option_count == 6
    assert por6.model.policy_option_hidden_film is True
    assert por6.model.policy_option_hidden_shift_weight == 0.22

    assert por8.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por8.model.variant == "por"
    assert por8.model.option_count == 8
    assert por8.model.policy_option_hidden_film is True
    assert por8.model.policy_option_hidden_shift_weight == 0.22


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


def test_load_memory_actor_hidden_scale_branch_gate_config() -> None:
    por = load_config(Path("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_scalemix95.yaml"))

    assert por.env.env_id == "MiniGrid-MemoryS9-v0"
    assert por.model.variant == "por"
    assert por.model.policy_option_hidden_film is True
    assert por.model.policy_option_hidden_use_duration_gate is True
    assert por.model.policy_option_hidden_branch_gates is True
    assert por.model.policy_option_hidden_scale_duration_mix == 0.95
    assert por.model.policy_option_hidden_shift_duration_mix == 1.0
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


def test_load_memory_actor_hidden_shiftgate075_scale325_adaptive_floor_configs() -> None:
    floor2875 = load_config(
        Path(
            "configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325_adaptive_floor2875.yaml"
        )
    )
    floor30 = load_config(
        Path(
            "configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325_adaptive_floor30.yaml"
        )
    )

    assert floor2875.model.policy_option_hidden_adaptive_scale_floor is True
    assert math.isclose(floor2875.model.policy_option_hidden_film_scale, 0.325)
    assert math.isclose(floor2875.model.policy_option_hidden_scale_floor, 0.2875)
    assert math.isclose(floor2875.model.policy_option_hidden_shift_gate_power, 0.75)

    assert floor30.model.policy_option_hidden_adaptive_scale_floor is True
    assert math.isclose(floor30.model.policy_option_hidden_film_scale, 0.325)
    assert math.isclose(floor30.model.policy_option_hidden_scale_floor, 0.30)
    assert math.isclose(floor30.model.policy_option_hidden_shift_gate_power, 0.75)


def test_load_memory_actor_hidden_shiftgate075_scale325_shift_floor_configs() -> None:
    floor20 = load_config(
        Path(
            "configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325_shiftfloor20.yaml"
        )
    )
    floor21 = load_config(
        Path(
            "configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325_shiftfloor21.yaml"
        )
    )

    assert floor20.model.policy_option_hidden_adaptive_shift_floor is True
    assert math.isclose(floor20.model.policy_option_hidden_film_scale, 0.325)
    assert math.isclose(floor20.model.policy_option_hidden_shift_floor, 0.20)
    assert math.isclose(floor20.model.policy_option_hidden_shift_gate_power, 0.75)

    assert floor21.model.policy_option_hidden_adaptive_shift_floor is True
    assert math.isclose(floor21.model.policy_option_hidden_film_scale, 0.325)
    assert math.isclose(floor21.model.policy_option_hidden_shift_floor, 0.21)
    assert math.isclose(floor21.model.policy_option_hidden_shift_gate_power, 0.75)


def test_load_memory_actor_hidden_shiftgate075_scale325_scale_branch_mix_configs() -> None:
    mix99 = load_config(
        Path(
            "configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325_scalemix99.yaml"
        )
    )
    mix975 = load_config(
        Path(
            "configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325_scalemix975.yaml"
        )
    )

    assert mix99.model.policy_option_hidden_branch_gates is True
    assert math.isclose(mix99.model.policy_option_hidden_film_scale, 0.325)
    assert math.isclose(mix99.model.policy_option_hidden_scale_duration_mix, 0.99)
    assert math.isclose(mix99.model.policy_option_hidden_shift_duration_mix, 1.0)
    assert math.isclose(mix99.model.policy_option_hidden_shift_gate_power, 0.75)

    assert mix975.model.policy_option_hidden_branch_gates is True
    assert math.isclose(mix975.model.policy_option_hidden_film_scale, 0.325)
    assert math.isclose(mix975.model.policy_option_hidden_scale_duration_mix, 0.975)
    assert math.isclose(mix975.model.policy_option_hidden_shift_duration_mix, 1.0)
    assert math.isclose(mix975.model.policy_option_hidden_shift_gate_power, 0.75)


def test_load_memory_actor_hidden_shiftgate075_scale325_powered_shift_floor_configs() -> None:
    power075 = load_config(
        Path(
            "configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325_shiftfloor20_power075.yaml"
        )
    )
    power05 = load_config(
        Path(
            "configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325_shiftfloor20_power05.yaml"
        )
    )

    assert power075.model.policy_option_hidden_adaptive_shift_floor is True
    assert math.isclose(power075.model.policy_option_hidden_film_scale, 0.325)
    assert math.isclose(power075.model.policy_option_hidden_shift_floor, 0.20)
    assert math.isclose(power075.model.policy_option_hidden_shift_floor_power, 0.75)
    assert math.isclose(power075.model.policy_option_hidden_shift_gate_power, 0.75)

    assert power05.model.policy_option_hidden_adaptive_shift_floor is True
    assert math.isclose(power05.model.policy_option_hidden_film_scale, 0.325)
    assert math.isclose(power05.model.policy_option_hidden_shift_floor, 0.20)
    assert math.isclose(power05.model.policy_option_hidden_shift_floor_power, 0.50)
    assert math.isclose(power05.model.policy_option_hidden_shift_gate_power, 0.75)
