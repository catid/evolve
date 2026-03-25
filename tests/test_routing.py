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


def test_sare_phase_memory_route_bias_keyed_residual_predictive_reports_predictive_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_predictive",
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
    assert "memory/keyed_route_bias_logits_norm" in output.metrics
    assert "memory/route_bias_logits_norm" in output.metrics
    assert "memory/route_memory_query_norm" in output.metrics
    assert "memory/predictive_hidden_norm" in output.metrics
    assert "memory/keyed_predictive_delta_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert float(output.metrics["memory/predictive_hidden_norm"]) >= 0.0
    assert float(output.metrics["memory/keyed_predictive_delta_norm"]) >= 0.0
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_predictive_blend_reports_predictive_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_predictive_blend",
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
    assert "memory/keyed_route_bias_logits_norm" in output.metrics
    assert "memory/route_bias_logits_norm" in output.metrics
    assert "memory/route_memory_query_norm" in output.metrics
    assert "memory/predictive_hidden_norm" in output.metrics
    assert "memory/predictive_blend_hidden_norm" in output.metrics
    assert "memory/keyed_predictive_delta_norm" in output.metrics
    assert "memory/keyed_predictive_blend" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert abs(float(output.metrics["memory/keyed_predictive_blend"]) - 0.5) < 1e-6
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_predictive_capped_reports_capped_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_predictive_capped",
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
    assert "memory/prior_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/predictive_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/capped_predictive_delta_logits_norm" in output.metrics
    assert "memory/keyed_predictive_delta_norm" in output.metrics
    assert "memory/predictive_delta_scale" in output.metrics
    assert "memory/keyed_predictive_cap_ratio" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert abs(float(output.metrics["memory/keyed_predictive_cap_ratio"]) - 0.5) < 1e-6
    assert 0.0 <= float(output.metrics["memory/predictive_delta_scale"]) <= 1.0
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_predictive_delta_mix_reports_mix_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_predictive_delta_mix",
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
    assert "memory/prior_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/predictive_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/mixed_predictive_delta_logits_norm" in output.metrics
    assert "memory/keyed_predictive_delta_norm" in output.metrics
    assert "memory/keyed_predictive_delta_mix" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert abs(float(output.metrics["memory/keyed_predictive_delta_mix"]) - 0.25) < 1e-6
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_predictive_positive_reports_positive_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_predictive_positive",
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
    assert "memory/prior_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/predictive_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/positive_predictive_delta_logits_norm" in output.metrics
    assert "memory/keyed_predictive_delta_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert float(output.metrics["memory/positive_predictive_delta_logits_norm"]) >= 0.0
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_predictive_aligned_reports_alignment_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_predictive_aligned",
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
    assert "memory/prior_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/predictive_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/aligned_predictive_delta_logits_norm" in output.metrics
    assert "memory/aligned_predictive_delta_fraction" in output.metrics
    assert "memory/keyed_predictive_delta_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert 0.0 <= float(output.metrics["memory/aligned_predictive_delta_fraction"]) <= 1.0
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_predictive_weak_prior_replace_reports_replacement_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_predictive_weak_prior_replace",
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
    assert "memory/prior_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/predictive_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/replaced_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/kept_prior_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/weak_prior_replace_gate" in output.metrics
    assert "memory/keyed_predictive_delta_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert 0.0 <= float(output.metrics["memory/weak_prior_replace_gate"]) <= 1.0
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_predictive_weak_prior_assist_reports_assist_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_predictive_weak_prior_assist",
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
    assert "memory/prior_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/predictive_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/assisted_predictive_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/weak_prior_assist_gate" in output.metrics
    assert "memory/keyed_predictive_delta_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert 0.0 <= float(output.metrics["memory/weak_prior_assist_gate"]) <= 1.0
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_predictive_weak_prior_top1_assist_reports_assist_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_predictive_weak_prior_top1_assist",
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
    assert "memory/prior_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/predictive_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/assisted_predictive_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/weak_prior_assist_gate" in output.metrics
    assert "memory/top1_weak_prior_assist_gate" in output.metrics
    assert "memory/top1_weak_prior_density" in output.metrics
    assert "memory/keyed_predictive_delta_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert 0.0 <= float(output.metrics["memory/weak_prior_assist_gate"]) <= 1.0
    assert 0.0 <= float(output.metrics["memory/top1_weak_prior_assist_gate"]) <= 1.0
    assert abs(float(output.metrics["memory/top1_weak_prior_density"]) - 0.25) < 1e-6
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_predictive_weak_prior_top1_max_merge_reports_merge_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_predictive_weak_prior_top1_max_merge",
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
    assert "memory/prior_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/predictive_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/maxmerged_predictive_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/weak_prior_merge_gate" in output.metrics
    assert "memory/top1_weak_prior_max_merge_gate" in output.metrics
    assert "memory/top1_weak_prior_density" in output.metrics
    assert "memory/keyed_predictive_delta_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert 0.0 <= float(output.metrics["memory/weak_prior_merge_gate"]) <= 1.0
    assert 0.0 <= float(output.metrics["memory/top1_weak_prior_max_merge_gate"]) <= 1.0
    assert abs(float(output.metrics["memory/top1_weak_prior_density"]) - 0.25) < 1e-6
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_predictive_top1_consensus_bonus_reports_consensus_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_predictive_top1_consensus_bonus",
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
    assert "memory/prior_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/predictive_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/consensus_bonus_logits_norm" in output.metrics
    assert "memory/top1_consensus_rate" in output.metrics
    assert "memory/top1_consensus_density" in output.metrics
    assert "memory/keyed_predictive_delta_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert float(output.metrics["memory/consensus_bonus_scale"]) == 0.5
    assert 0.0 <= float(output.metrics["memory/top1_consensus_rate"]) <= 1.0
    assert 0.0 <= float(output.metrics["memory/top1_consensus_density"]) <= 0.25
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_predictive_top1_disagreement_bonus_reports_disagreement_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_predictive_top1_disagreement_bonus",
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
    assert "memory/prior_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/predictive_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/disagreement_bonus_logits_norm" in output.metrics
    assert "memory/top1_disagreement_rate" in output.metrics
    assert "memory/top1_disagreement_density" in output.metrics
    assert "memory/keyed_predictive_delta_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert float(output.metrics["memory/disagreement_bonus_scale"]) == 0.5
    assert 0.0 <= float(output.metrics["memory/top1_disagreement_rate"]) <= 1.0
    assert 0.0 <= float(output.metrics["memory/top1_disagreement_density"]) <= 0.25
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_predictive_base_top1_bonus_reports_base_bonus_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_predictive_base_top1_bonus",
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
    assert "memory/predictive_base_route_bias_logits_norm" in output.metrics
    assert "memory/predictive_base_bonus_logits_norm" in output.metrics
    assert "memory/predictive_base_top1_disagreement_rate" in output.metrics
    assert "memory/predictive_base_top1_density" in output.metrics
    assert "memory/predictive_hidden_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert float(output.metrics["memory/predictive_base_bonus_scale"]) == 0.5
    assert 0.0 <= float(output.metrics["memory/predictive_base_top1_disagreement_rate"]) <= 1.0
    assert abs(float(output.metrics["memory/predictive_base_top1_density"]) - 0.25) < 1e-6
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_predictive_base_delta_mix_reports_base_delta_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_predictive_base_delta_mix",
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
    assert "memory/predictive_base_route_bias_logits_norm" in output.metrics
    assert "memory/predictive_base_delta_logits_norm" in output.metrics
    assert "memory/mixed_predictive_base_delta_logits_norm" in output.metrics
    assert "memory/keyed_route_bias_logits_norm" in output.metrics
    assert "memory/predictive_hidden_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert float(output.metrics["memory/predictive_base_delta_mix"]) == 0.25
    assert float(output.metrics["memory/predictive_base_delta_logits_norm"]) >= 0.0
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_predictive_base_delta_entropy_gate_reports_entropy_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_predictive_base_delta_entropy_gate",
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
    assert "memory/predictive_base_route_bias_logits_norm" in output.metrics
    assert "memory/predictive_base_delta_logits_norm" in output.metrics
    assert "memory/mixed_predictive_base_delta_logits_norm" in output.metrics
    assert "memory/predictive_base_uncertainty_gate" in output.metrics
    assert "memory/predictive_hidden_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert float(output.metrics["memory/predictive_base_delta_mix"]) == 0.25
    assert 0.0 <= float(output.metrics["memory/predictive_base_uncertainty_gate"]) <= 1.0
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_predictive_base_delta_weak_base_gate_reports_gate_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_predictive_base_delta_weak_base_gate",
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
    assert "memory/predictive_base_route_bias_logits_norm" in output.metrics
    assert "memory/predictive_base_delta_logits_norm" in output.metrics
    assert "memory/mixed_predictive_base_delta_logits_norm" in output.metrics
    assert "memory/predictive_base_weak_gate" in output.metrics
    assert "memory/predictive_base_target_base_norm" in output.metrics
    assert "memory/predictive_hidden_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert float(output.metrics["memory/predictive_base_delta_mix"]) == 0.25
    assert abs(float(output.metrics["memory/predictive_base_target_base_norm"]) - 1.0) < 1e-6
    assert 0.0 <= float(output.metrics["memory/predictive_base_weak_gate"]) <= 1.0
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_predictive_delta_gate_reports_gate_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_predictive_delta_gate",
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
    assert "memory/prior_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/predictive_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/positive_predictive_delta_logits_norm" in output.metrics
    assert "memory/gated_positive_predictive_delta_logits_norm" in output.metrics
    assert "memory/predictive_delta_gate" in output.metrics
    assert "memory/keyed_predictive_delta_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert 0.0 <= float(output.metrics["memory/predictive_delta_gate"]) <= 1.0
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_ema_reports_ema_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_ema",
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
    assert "ema_hidden" in output.next_state
    assert output.next_state["hidden"].shape == (1, 128)
    assert output.next_state["ema_hidden"].shape == (1, 128)
    assert "memory/base_route_bias_logits_norm" in output.metrics
    assert "memory/keyed_route_bias_logits_norm" in output.metrics
    assert "memory/route_bias_logits_norm" in output.metrics
    assert "memory/route_memory_query_norm" in output.metrics
    assert "memory/ema_hidden_norm" in output.metrics
    assert "memory/keyed_ema_delta_norm" in output.metrics
    assert "memory/keyed_ema_decay" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert abs(float(output.metrics["memory/keyed_ema_decay"]) - 0.75) < 1e-6
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_margin_gate_reports_gate_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_margin_gate",
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
    assert "memory/base_route_top2_margin" in output.metrics
    assert "memory/base_route_top2_margin_normalized" in output.metrics
    assert "memory/keyed_residual_gate_mean" in output.metrics
    assert "memory/gated_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/keyed_residual_boost" in output.metrics
    assert "memory/route_bias_logits_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert 1.0 <= float(output.metrics["memory/keyed_residual_gate_mean"]) <= 1.5
    assert float(output.metrics["memory/keyed_residual_boost"]) == 0.5
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_mild_boost_reports_boost_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_mild_boost",
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
    assert "memory/keyed_route_bias_logits_norm" in output.metrics
    assert "memory/boosted_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/keyed_residual_scale" in output.metrics
    assert "memory/route_bias_logits_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert abs(float(output.metrics["memory/keyed_residual_scale"]) - 1.5) < 1e-6
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_challenger_reports_disagreement_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_challenger",
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
    assert "memory/challenger_route_bias_logits_norm" in output.metrics
    assert "memory/base_topk_density" in output.metrics
    assert "memory/keyed_topk_density" in output.metrics
    assert "memory/challenger_mask_density" in output.metrics
    assert "memory/challenger_scale" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert float(output.metrics["memory/challenger_scale"]) == 0.5
    assert 0.0 <= float(output.metrics["memory/challenger_mask_density"]) <= 1.0
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_base_clip_reports_clip_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_base_clip",
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
    assert "memory/clipped_base_route_bias_logits_norm" in output.metrics
    assert "memory/base_clip_scale_mean" in output.metrics
    assert "memory/base_clip_scale_min" in output.metrics
    assert "memory/base_clip_scale_max" in output.metrics
    assert "memory/base_clip_target_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert float(output.metrics["memory/base_clip_target_norm"]) == 0.8
    assert 0.0 < float(output.metrics["memory/base_clip_scale_mean"]) <= 1.0
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_ratio_gate_reports_gate_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_ratio_gate",
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
    assert "memory/gated_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/base_keyed_ratio_mean" in output.metrics
    assert "memory/ratio_gate_mean" in output.metrics
    assert "memory/ratio_gate_min" in output.metrics
    assert "memory/ratio_gate_max" in output.metrics
    assert "memory/keyed_residual_ratio_threshold" in output.metrics
    assert "memory/keyed_residual_max_boost" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert float(output.metrics["memory/keyed_residual_ratio_threshold"]) == 20.0
    assert float(output.metrics["memory/keyed_residual_max_boost"]) == 1.5
    assert 0.0 <= float(output.metrics["memory/ratio_gate_mean"]) <= 1.0
    assert 1.0 <= float(output.metrics["memory/keyed_residual_scale_mean"]) <= 1.5
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_orthogonal_reports_alignment_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_orthogonal",
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
    assert "memory/aligned_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/orthogonal_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/keyed_route_alignment_cosine" in output.metrics
    assert "memory/route_bias_logits_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert -1.0 <= float(output.metrics["memory/keyed_route_alignment_cosine"]) <= 1.0
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_floor_boost_reports_boost_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_floor_boost",
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
    assert "memory/boosted_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/keyed_residual_scale_mean" in output.metrics
    assert "memory/keyed_residual_scale_min" in output.metrics
    assert "memory/keyed_residual_scale_max" in output.metrics
    assert "memory/keyed_residual_target_floor" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert float(output.metrics["memory/keyed_residual_target_floor"]) == 0.05
    assert 1.0 <= float(output.metrics["memory/keyed_residual_scale_mean"]) <= 1.75
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_target_norm_reports_scale_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_target_norm",
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
    assert "memory/scaled_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/keyed_residual_scale_mean" in output.metrics
    assert "memory/keyed_residual_scale_min" in output.metrics
    assert "memory/keyed_residual_scale_max" in output.metrics
    assert "memory/keyed_residual_target_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert float(output.metrics["memory/keyed_residual_target_norm"]) == 0.05
    assert 0.75 <= float(output.metrics["memory/keyed_residual_scale_mean"]) <= 1.75
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_ratio_matched_reports_ratio_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_ratio_matched",
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
    assert "memory/keyed_route_bias_logits_norm" in output.metrics
    assert "memory/scaled_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/keyed_residual_scale_mean" in output.metrics
    assert "memory/keyed_residual_scale_min" in output.metrics
    assert "memory/keyed_residual_scale_max" in output.metrics
    assert "memory/keyed_residual_target_ratio" in output.metrics
    assert "memory/route_bias_logits_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert 0.5 <= float(output.metrics["memory/keyed_residual_scale_mean"]) <= 3.0
    assert float(output.metrics["memory/keyed_residual_target_ratio"]) == 0.1
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_per_expert_scale_reports_scale_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_per_expert_scale",
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
    assert "memory/keyed_route_bias_logits_norm" in output.metrics
    assert "memory/scaled_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/keyed_residual_scale_mean" in output.metrics
    assert "memory/keyed_residual_scale_std" in output.metrics
    assert "memory/route_bias_logits_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert abs(float(output.metrics["memory/keyed_residual_scale_mean"]) - 1.0) < 1e-6
    assert abs(float(output.metrics["memory/keyed_residual_scale_std"]) - 0.0) < 1e-6
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_learned_scale_reports_scale_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_learned_scale",
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
    assert "memory/keyed_route_bias_logits_norm" in output.metrics
    assert "memory/scaled_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/keyed_residual_scale" in output.metrics
    assert "memory/route_bias_logits_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert abs(float(output.metrics["memory/keyed_residual_scale"]) - 1.0) < 1e-6
    env.close()


def test_sare_phase_memory_route_bias_keyed_residual_topk_masked_reports_mask_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_topk_masked",
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
    assert "memory/keyed_route_bias_logits_norm" in output.metrics
    assert "memory/masked_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/masked_keyed_route_bias_density" in output.metrics
    assert "memory/route_bias_logits_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert abs(float(output.metrics["memory/masked_keyed_route_bias_density"]) - 0.5) < 1e-6
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


def test_sare_phase_memory_route_bias_keyed_residual_hidden_gate_reports_gate_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(
        ModelConfig(
            variant="sare_phase_memory_route_bias_keyed_residual_hidden_gate",
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
    assert "memory/keyed_residual_hidden_gate_mean" in output.metrics
    assert "memory/keyed_residual_hidden_gate_std" in output.metrics
    assert "memory/keyed_residual_hidden_gate_min" in output.metrics
    assert "memory/keyed_residual_hidden_gate_max" in output.metrics
    assert "memory/gated_keyed_route_bias_logits_norm" in output.metrics
    assert "memory/route_bias_logits_norm" in output.metrics
    assert float(output.metrics["memory/route_bias_scale"]) == 0.5
    assert 0.5 <= float(output.metrics["memory/keyed_residual_hidden_gate_mean"]) <= 1.5
    assert 0.5 <= float(output.metrics["memory/keyed_residual_hidden_gate_min"]) <= 1.5
    assert 0.5 <= float(output.metrics["memory/keyed_residual_hidden_gate_max"]) <= 1.5
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
