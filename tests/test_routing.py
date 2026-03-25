import gymnasium as gym
import minigrid  # noqa: F401
import torch

from psmn_rl.config import ModelConfig
from psmn_rl.models.factory import build_model


def _obs() -> tuple[gym.Env, dict[str, torch.Tensor], torch.Tensor]:
    env = gym.make("MiniGrid-DoorKey-5x5-v0")
    obs, _ = env.reset(seed=0)
    obs_t = {
        "image": torch.as_tensor(obs["image"]).unsqueeze(0),
        "direction": torch.as_tensor(obs["direction"]).unsqueeze(0),
    }
    done = torch.ones(1, dtype=torch.bool)
    return env, obs_t, done


def test_sare_reports_route_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(ModelConfig(variant="sare", expert_count=4, top_k=2), env.observation_space, env.action_space)
    output = model.forward(obs, state={}, done=done)
    assert "route_entropy" in output.metrics
    assert "path_coverage_top_10" in output.metrics
    assert any(key.startswith("expert_load_") for key in output.metrics)
    env.close()


def test_sare_phase_memory_gated_reports_memory_gate_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_gated",
            expert_count=4,
            top_k=2,
            memory_mix=0.5,
            memory_gate_bias=0.0,
            memory_reset_bias=-2.0,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "hidden" in output.next_state
    assert output.next_state["hidden"].shape == (1, 128)
    assert 0.0 <= float(output.metrics["memory/update_gate_mean"]) <= 1.0
    assert 0.0 <= float(output.metrics["memory/reset_gate_mean"]) <= 1.0
    assert "memory/route_context_norm" in output.metrics
    env.close()


def test_sare_phase_memory_route_bias_reports_route_memory_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias",
            expert_count=4,
            top_k=2,
            memory_mix=0.5,
            route_memory_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "hidden" in output.next_state
    assert output.next_state["hidden"].shape == (1, 128)
    assert "memory/route_bias_norm" in output.metrics
    assert "memory/route_bias_absmax" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    env.close()


def test_sare_phase_memory_route_bias_keyed_reports_key_alignment_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed",
            expert_count=4,
            top_k=2,
            memory_mix=0.5,
            route_memory_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "hidden" in output.next_state
    assert output.next_state["hidden"].shape == (1, 128)
    assert "memory/route_bias_norm" in output.metrics
    assert "memory/route_memory_query_norm" in output.metrics
    assert "memory/route_bias_logits_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert float(output.metrics["memory/route_memory_query_norm"]) >= 0.0
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_reports_residual_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual",
            expert_count=4,
            top_k=2,
            memory_mix=0.5,
            route_memory_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "hidden" in output.next_state
    assert output.next_state["hidden"].shape == (1, 128)
    assert "memory/route_bias_norm" in output.metrics
    assert "memory/base_route_bias_logits_norm" in output.metrics
    assert "memory/keyed_route_bias_logits_norm" in output.metrics
    assert "memory/route_bias_logits_norm" in output.metrics
    assert "memory/route_memory_query_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_postact_reports_postact_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_postact",
            expert_count=4,
            top_k=2,
            memory_mix=0.5,
            route_memory_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "hidden" in output.next_state
    assert output.next_state["hidden"].shape == (1, 128)
    assert "memory/base_route_bias_logits_norm" in output.metrics
    assert "memory/base_route_bias_norm" in output.metrics
    assert "memory/keyed_route_bias_logits_norm" in output.metrics
    assert "memory/keyed_route_bias_norm" in output.metrics
    assert "memory/post_activation_route_bias_norm" in output.metrics
    assert "memory/route_memory_query_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_centered_reports_centering_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_centered",
            expert_count=4,
            top_k=2,
            memory_mix=0.5,
            route_memory_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "hidden" in output.next_state
    assert output.next_state["hidden"].shape == (1, 128)
    assert "memory/keyed_route_bias_logits_norm" in output.metrics
    assert "memory/centered_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/centered_keyed_route_bias_absmean" in output.metrics
    assert "memory/route_bias_logits_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert float(output.metrics["memory/centered_keyed_route_bias_absmean"]) < 1e-5
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_boosted_reports_boost_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_boosted",
            expert_count=4,
            top_k=2,
            memory_mix=0.5,
            route_memory_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "hidden" in output.next_state
    assert output.next_state["hidden"].shape == (1, 128)
    assert "memory/route_bias_norm" in output.metrics
    assert "memory/base_route_bias_logits_norm" in output.metrics
    assert "memory/keyed_route_bias_logits_norm" in output.metrics
    assert "memory/boosted_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/route_bias_logits_norm" in output.metrics
    assert "memory/keyed_residual_scale" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert float(output.metrics["memory/keyed_residual_scale"]) == 4.0
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_entropy_gate_reports_gate_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_entropy_gate",
            expert_count=4,
            top_k=2,
            memory_mix=0.5,
            route_memory_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "hidden" in output.next_state
    assert output.next_state["hidden"].shape == (1, 128)
    assert "memory/base_route_bias_entropy" in output.metrics
    assert "memory/keyed_residual_gate_mean" in output.metrics
    assert "memory/gated_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/route_bias_logits_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert 0.5 <= float(output.metrics["memory/keyed_residual_gate_mean"]) <= 1.5
    env.close()


def test_sare_phase_memory_route_bias_contextual_reports_context_bias_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_contextual",
            expert_count=4,
            top_k=2,
            memory_mix=0.5,
            route_memory_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "hidden" in output.next_state
    assert output.next_state["hidden"].shape == (1, 128)
    assert "memory/route_bias_norm" in output.metrics
    assert "memory/route_context_bias_norm" in output.metrics
    assert "memory/route_memory_bias_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    env.close()


def test_sare_phase_memory_route_bias_scale_gate_reports_scale_gate_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_scale_gate",
            expert_count=4,
            top_k=2,
            memory_mix=0.5,
            route_memory_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "hidden" in output.next_state
    assert output.next_state["hidden"].shape == (1, 128)
    assert "memory/route_bias_norm" in output.metrics
    assert "memory/route_scale_gate_mean" in output.metrics
    assert "memory/base_route_bias_norm" in output.metrics
    assert 0.0 <= float(output.metrics["memory/route_scale_gate_mean"]) <= 1.0
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    env.close()


def test_sare_phase_memory_route_bias_normalized_reports_normalized_bias_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_normalized",
            expert_count=4,
            top_k=2,
            memory_mix=0.5,
            route_memory_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "hidden" in output.next_state
    assert output.next_state["hidden"].shape == (1, 128)
    assert "memory/route_bias_norm" in output.metrics
    assert "memory/base_route_bias_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert float(output.metrics["memory/route_bias_norm"]) > 0.0
    env.close()


def test_sare_phase_memory_route_bias_layernorm_reports_layernorm_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_layernorm",
            expert_count=4,
            top_k=2,
            memory_mix=0.5,
            route_memory_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "hidden" in output.next_state
    assert output.next_state["hidden"].shape == (1, 128)
    assert "memory/route_bias_norm" in output.metrics
    assert "memory/route_hidden_prenorm_norm" in output.metrics
    assert "memory/route_hidden_postnorm_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert float(output.metrics["memory/route_hidden_postnorm_norm"]) >= 0.0
    env.close()


def test_sare_phase_memory_route_bias_gated_reports_hybrid_memory_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_gated",
            expert_count=4,
            top_k=2,
            memory_mix=0.5,
            memory_gate_bias=0.0,
            memory_reset_bias=-2.0,
            route_memory_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "hidden" in output.next_state
    assert output.next_state["hidden"].shape == (1, 128)
    assert "memory/route_bias_norm" in output.metrics
    assert "memory/update_gate_mean" in output.metrics
    assert "memory/reset_gate_mean" in output.metrics
    assert 0.0 <= float(output.metrics["memory/update_gate_mean"]) <= 1.0
    assert 0.0 <= float(output.metrics["memory/reset_gate_mean"]) <= 1.0
    env.close()


def test_sare_phase_memory_route_bias_dual_reports_dual_memory_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_dual",
            expert_count=4,
            top_k=2,
            memory_mix=0.5,
            route_memory_scale=0.5,
        ),
        env.observation_space,
        env.action_space,
    )
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "hidden" in output.next_state
    assert "route_hidden" in output.next_state
    assert output.next_state["hidden"].shape == (1, 128)
    assert output.next_state["route_hidden"].shape == (1, 128)
    assert "memory/route_bias_norm" in output.metrics
    assert "memory/route_hidden_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    env.close()


def test_treg_h_reports_hop_and_ponder_metrics() -> None:
    env, obs, done = _obs()
    model = build_model(ModelConfig(variant="treg_h", expert_count=4), env.observation_space, env.action_space)
    output = model.forward(obs, state={}, done=done)
    assert "avg_hop_count" in output.metrics
    assert "avg_halting_probability" in output.metrics
    assert "ponder_loss" in output.aux_losses
    env.close()


def test_srw_reports_relational_usage() -> None:
    env, obs, done = _obs()
    model = build_model(ModelConfig(variant="srw", expert_count=4), env.observation_space, env.action_space)
    output = model.forward(obs, state={}, done=done)
    assert "relational_usage_rate" in output.metrics
    env.close()


def test_por_updates_option_state() -> None:
    env, obs, done = _obs()
    model = build_model(ModelConfig(variant="por", option_count=4), env.observation_space, env.action_space)
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "option_probs" in output.next_state
    assert output.next_state["option_probs"].shape == (1, 4)
    assert "option_duration" in output.metrics
    env.close()
