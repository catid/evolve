import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch

from psmn_rl.config import ModelConfig
from psmn_rl.models.factory import build_model


ALL_VARIANTS = [
    "flat_dense",
    "token_dense",
    "token_gru",
    "single_expert",
    "sare",
    "sare_phase_memory",
    "sare_phase_memory_gated",
    "sare_phase_memory_route_bias",
    "sare_phase_memory_route_bias_keyed",
    "sare_phase_memory_route_bias_keyed_residual",
    "sare_phase_memory_route_bias_keyed_residual_predictive",
    "sare_phase_memory_route_bias_keyed_residual_predictive_blend",
    "sare_phase_memory_route_bias_keyed_residual_predictive_capped",
    "sare_phase_memory_route_bias_keyed_residual_predictive_delta_mix",
    "sare_phase_memory_route_bias_keyed_residual_predictive_positive",
    "sare_phase_memory_route_bias_keyed_residual_predictive_aligned",
    "sare_phase_memory_route_bias_keyed_residual_predictive_weak_prior_replace",
    "sare_phase_memory_route_bias_keyed_residual_predictive_weak_prior_assist",
    "sare_phase_memory_route_bias_keyed_residual_predictive_weak_prior_top1_assist",
    "sare_phase_memory_route_bias_keyed_residual_predictive_weak_prior_top1_max_merge",
    "sare_phase_memory_route_bias_keyed_residual_predictive_top1_consensus_bonus",
    "sare_phase_memory_route_bias_keyed_residual_predictive_top1_disagreement_bonus",
    "sare_phase_memory_route_bias_keyed_residual_predictive_base_top1_bonus",
    "sare_phase_memory_route_bias_keyed_residual_predictive_base_delta_mix",
    "sare_phase_memory_route_bias_keyed_residual_predictive_base_delta_entropy_gate",
    "sare_phase_memory_route_bias_keyed_residual_predictive_base_delta_weak_base_gate",
    "sare_phase_memory_route_bias_keyed_residual_predictive_delta_gate",
    "sare_phase_memory_route_bias_keyed_residual_ema",
    "sare_phase_memory_route_bias_keyed_residual_margin_gate",
    "sare_phase_memory_route_bias_keyed_residual_mild_boost",
    "sare_phase_memory_route_bias_keyed_residual_challenger",
    "sare_phase_memory_route_bias_keyed_residual_base_clip",
    "sare_phase_memory_route_bias_keyed_residual_ratio_gate",
    "sare_phase_memory_route_bias_keyed_residual_orthogonal",
    "sare_phase_memory_route_bias_keyed_residual_floor_boost",
    "sare_phase_memory_route_bias_keyed_residual_target_norm",
    "sare_phase_memory_route_bias_keyed_residual_ratio_matched",
    "sare_phase_memory_route_bias_keyed_residual_per_expert_scale",
    "sare_phase_memory_route_bias_keyed_residual_learned_scale",
    "sare_phase_memory_route_bias_keyed_residual_topk_masked",
    "sare_phase_memory_route_bias_keyed_residual_postact",
    "sare_phase_memory_route_bias_keyed_residual_centered",
    "sare_phase_memory_route_bias_keyed_residual_boosted",
    "sare_phase_memory_route_bias_keyed_residual_entropy_gate",
    "sare_phase_memory_route_bias_keyed_residual_hidden_gate",
    "sare_phase_memory_route_bias_keyed_residual_weak_base_expert_gate",
    "sare_phase_memory_route_bias_keyed_residual_weak_base_expert_concentration_blend",
    "sare_phase_memory_route_bias_keyed_residual_weak_base_margin_advantage_expert_gate",
    "sare_phase_memory_route_bias_keyed_residual_weak_base_margin_bonus_expert_gate",
    "sare_phase_memory_route_bias_keyed_residual_weak_base_sign_disagreement_expert_gate",
    "sare_phase_memory_route_bias_keyed_residual_weak_base_cosine_disagreement_expert_gate",
    "sare_phase_memory_route_bias_keyed_residual_weak_base_expert_softmax_redistribute",
    "sare_phase_memory_route_bias_keyed_residual_weak_base_keyed_relative_expert_gate",
    "sare_phase_memory_route_bias_keyed_residual_weak_base_sign_aligned_expert_gate",
    "sare_phase_memory_route_bias_keyed_residual_weak_base_keyed_top2_expert_gate",
    "sare_phase_memory_route_bias_keyed_residual_weak_base_keyed_abs_top2_expert_gate",
    "sare_phase_memory_route_bias_keyed_residual_weak_base_expert_high_base_gate",
    "sare_phase_memory_route_bias_keyed_residual_weak_base_expert_high_base_floor_gate",
    "sare_phase_memory_route_bias_keyed_residual_weak_base_expert_low_base_renorm",
    "sare_phase_memory_route_bias_keyed_residual_weak_base_expert_renorm",
    "sare_phase_memory_route_bias_keyed_residual_weak_base_weak_keyed_expert_gate",
    "sare_phase_memory_route_bias_contextual",
    "sare_phase_memory_route_bias_scale_gate",
    "sare_phase_memory_route_bias_normalized",
    "sare_phase_memory_route_bias_layernorm",
    "sare_phase_memory_route_bias_gated",
    "sare_phase_memory_route_bias_dual",
    "treg_h",
    "srw",
    "por",
]


def _sample_obs(env: gym.Env) -> dict[str, torch.Tensor]:
    obs, _ = env.reset(seed=0)
    return {
        "image": torch.as_tensor(obs["image"]).unsqueeze(0),
        "direction": torch.as_tensor(obs["direction"]).unsqueeze(0),
    }


def test_all_variants_forward_shapes() -> None:
    env = gym.make("MiniGrid-DoorKey-5x5-v0")
    obs = _sample_obs(env)
    done = torch.ones(1, dtype=torch.bool)
    for variant in ALL_VARIANTS:
        model_config = ModelConfig(variant=variant)
        model = build_model(model_config, env.observation_space, env.action_space)
        state = model.initial_state(batch_size=1, device=torch.device("cpu"))
        output = model.forward(obs, state=state, done=done)
        assert output.logits.shape == (1, env.action_space.n)
        assert output.value.shape == (1,)
    env.close()


def test_visual_box_variants_forward_shapes() -> None:
    observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    action_space = gym.spaces.Discrete(15)
    obs = {"pixels": torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.uint8)}
    done = torch.ones(1, dtype=torch.bool)
    for variant in ALL_VARIANTS:
        model_config = ModelConfig(variant=variant, hidden_size=64, token_dim=64, patch_size=8)
        model = build_model(model_config, observation_space, action_space)
        state = model.initial_state(batch_size=1, device=torch.device("cpu"))
        output = model.forward(obs, state=state, done=done)
        assert output.logits.shape == (1, action_space.n)
        assert output.value.shape == (1,)


def test_margin_residual_policy_head_reports_decode_metrics() -> None:
    env = gym.make("MiniGrid-MemoryS9-v0")
    obs = _sample_obs(env)
    done = torch.ones(1, dtype=torch.bool)
    model = build_model(
        ModelConfig(
            variant="por",
            policy_margin_residual=True,
            policy_margin_threshold=0.25,
            policy_margin_sharpness=12.0,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "policy/margin_residual_gate_mean" in output.metrics
    assert "policy/margin_residual_low_margin_mean" in output.metrics
    assert "policy/margin_residual_base_margin" in output.metrics
    assert "policy/margin_residual_logits_norm" in output.metrics
    env.close()


def test_por_option_action_adapter_reports_metrics() -> None:
    env = gym.make("MiniGrid-MemoryS9-v0")
    obs = _sample_obs(env)
    done = torch.ones(1, dtype=torch.bool)
    model = build_model(
        ModelConfig(
            variant="por",
            por_option_action_adapter=True,
            por_option_action_adapter_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "policy/option_action_adapter_stability" in output.metrics
    assert "policy/option_action_adapter_bias_norm" in output.metrics
    env.close()


def test_por_option_hidden_residual_reports_metrics() -> None:
    env = gym.make("MiniGrid-MemoryS9-v0")
    obs = _sample_obs(env)
    done = torch.ones(1, dtype=torch.bool)
    model = build_model(
        ModelConfig(
            variant="por",
            por_option_hidden_residual=True,
            por_option_hidden_residual_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "policy/option_hidden_residual_stability" in output.metrics
    assert "policy/option_hidden_residual_norm" in output.metrics
    env.close()


def test_por_option_action_experts_report_metrics() -> None:
    env = gym.make("MiniGrid-MemoryS9-v0")
    obs = _sample_obs(env)
    done = torch.ones(1, dtype=torch.bool)
    model = build_model(
        ModelConfig(
            variant="por",
            por_option_action_experts=True,
            por_option_action_experts_scale=1.0,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "policy/option_action_experts_stability" in output.metrics
    assert "policy/option_action_experts_bias_norm" in output.metrics
    env.close()


def test_por_option_film_reports_metrics() -> None:
    env = gym.make("MiniGrid-MemoryS9-v0")
    obs = _sample_obs(env)
    done = torch.ones(1, dtype=torch.bool)
    model = build_model(
        ModelConfig(
            variant="por",
            por_option_film=True,
            por_option_film_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "policy/option_film_stability" in output.metrics
    assert "policy/option_film_scale_norm" in output.metrics
    env.close()


def test_por_option_context_logits_reports_metrics() -> None:
    env = gym.make("MiniGrid-MemoryS9-v0")
    obs = _sample_obs(env)
    done = torch.ones(1, dtype=torch.bool)
    model = build_model(
        ModelConfig(
            variant="por",
            por_option_context_logits=True,
            por_option_context_logits_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "policy/option_context_logits_stability" in output.metrics
    assert "policy/option_context_logits_bias_norm" in output.metrics
    env.close()


def test_por_option_margin_adapter_reports_metrics() -> None:
    env = gym.make("MiniGrid-MemoryS9-v0")
    obs = _sample_obs(env)
    done = torch.ones(1, dtype=torch.bool)
    model = build_model(
        ModelConfig(
            variant="por",
            por_option_margin_adapter=True,
            policy_option_margin_adapter=True,
            policy_option_margin_adapter_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "policy/option_margin_adapter_stability" in output.metrics
    assert "policy/option_margin_adapter_gate_mean" in output.metrics
    assert "policy/option_margin_adapter_effective_norm" in output.metrics
    env.close()


def test_por_option_top2_rerank_reports_metrics() -> None:
    env = gym.make("MiniGrid-MemoryS9-v0")
    obs = _sample_obs(env)
    done = torch.ones(1, dtype=torch.bool)
    model = build_model(
        ModelConfig(
            variant="por",
            por_option_top2_rerank=True,
            policy_option_top2_rerank=True,
            policy_option_top2_rerank_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "policy/option_top2_rerank_stability" in output.metrics
    assert "policy/option_top2_rerank_gate_mean" in output.metrics
    assert "policy/option_top2_rerank_effective_delta_mean_abs" in output.metrics
    env.close()


def test_por_option_top2_duration_reports_metrics() -> None:
    env = gym.make("MiniGrid-MemoryS9-v0")
    obs = _sample_obs(env)
    done = torch.ones(1, dtype=torch.bool)
    model = build_model(
        ModelConfig(
            variant="por",
            por_option_top2_rerank=True,
            policy_option_top2_rerank=True,
            policy_option_top2_use_duration_gate=True,
            policy_option_top2_rerank_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "policy/option_top2_rerank_gate_signal_mean" in output.metrics
    assert "policy/option_top2_rerank_gate_mean" in output.metrics
    assert "policy/option_top2_rerank_effective_delta_mean_abs" in output.metrics
    env.close()


def test_por_actor_hidden_film_reports_metrics() -> None:
    env = gym.make("MiniGrid-MemoryS9-v0")
    obs = _sample_obs(env)
    done = torch.ones(1, dtype=torch.bool)
    model = build_model(
        ModelConfig(
            variant="por",
            por_option_actor_features=True,
            policy_option_hidden_film=True,
            policy_option_hidden_film_scale=0.5,
            policy_option_hidden_use_duration_gate=True,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "policy/option_actor_features_duration_gate" in output.metrics
    assert "policy/option_hidden_film_gate_signal_mean" in output.metrics
    assert "policy/option_hidden_film_scale_norm" in output.metrics
    assert "policy/option_hidden_film_shift_norm" in output.metrics
    env.close()


def test_logit_gain_reports_metrics() -> None:
    env = gym.make("MiniGrid-MemoryS9-v0")
    obs = _sample_obs(env)
    done = torch.ones(1, dtype=torch.bool)
    model = build_model(
        ModelConfig(
            variant="por",
            policy_logit_gain=True,
            policy_logit_gain_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "policy/logit_gain_gate_mean" in output.metrics
    assert "policy/logit_gain_raw_norm" in output.metrics
    env.close()


def test_top2_rerank_reports_metrics() -> None:
    env = gym.make("MiniGrid-MemoryS9-v0")
    obs = _sample_obs(env)
    done = torch.ones(1, dtype=torch.bool)
    model = build_model(
        ModelConfig(
            variant="por",
            policy_top2_rerank=True,
            policy_top2_rerank_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "policy/top2_rerank_gate_mean" in output.metrics
    assert "policy/top2_rerank_raw_delta_mean_abs" in output.metrics
    env.close()


def test_por_option_context_gain_reports_metrics() -> None:
    env = gym.make("MiniGrid-MemoryS9-v0")
    obs = _sample_obs(env)
    done = torch.ones(1, dtype=torch.bool)
    model = build_model(
        ModelConfig(
            variant="por",
            por_option_context_gain=True,
            por_option_context_gain_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "policy/option_context_gain_stability" in output.metrics
    assert "policy/option_context_gain_mean" in output.metrics
    env.close()


def test_por_option_context_film_reports_metrics() -> None:
    env = gym.make("MiniGrid-MemoryS9-v0")
    obs = _sample_obs(env)
    done = torch.ones(1, dtype=torch.bool)
    model = build_model(
        ModelConfig(
            variant="por",
            por_option_context_film=True,
            por_option_context_film_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "policy/option_context_film_stability" in output.metrics
    assert "policy/option_context_film_scale_norm" in output.metrics
    env.close()
