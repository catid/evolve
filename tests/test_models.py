import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch

from psmn_rl.config import ModelConfig
from psmn_rl.models.factory import build_model


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
    variants = ["flat_dense", "token_dense", "token_gru", "single_expert", "sare", "sare_phase_memory", "sare_phase_memory_gated", "sare_phase_memory_route_bias", "sare_phase_memory_route_bias_keyed", "sare_phase_memory_route_bias_keyed_residual", "sare_phase_memory_route_bias_keyed_residual_predictive", "sare_phase_memory_route_bias_keyed_residual_predictive_blend", "sare_phase_memory_route_bias_keyed_residual_predictive_capped", "sare_phase_memory_route_bias_keyed_residual_predictive_delta_mix", "sare_phase_memory_route_bias_keyed_residual_predictive_positive", "sare_phase_memory_route_bias_keyed_residual_predictive_aligned", "sare_phase_memory_route_bias_keyed_residual_predictive_weak_prior_replace", "sare_phase_memory_route_bias_keyed_residual_predictive_weak_prior_assist", "sare_phase_memory_route_bias_keyed_residual_predictive_weak_prior_top1_assist", "sare_phase_memory_route_bias_keyed_residual_predictive_weak_prior_top1_max_merge", "sare_phase_memory_route_bias_keyed_residual_predictive_top1_consensus_bonus", "sare_phase_memory_route_bias_keyed_residual_predictive_delta_gate", "sare_phase_memory_route_bias_keyed_residual_ema", "sare_phase_memory_route_bias_keyed_residual_margin_gate", "sare_phase_memory_route_bias_keyed_residual_mild_boost", "sare_phase_memory_route_bias_keyed_residual_challenger", "sare_phase_memory_route_bias_keyed_residual_base_clip", "sare_phase_memory_route_bias_keyed_residual_ratio_gate", "sare_phase_memory_route_bias_keyed_residual_orthogonal", "sare_phase_memory_route_bias_keyed_residual_floor_boost", "sare_phase_memory_route_bias_keyed_residual_target_norm", "sare_phase_memory_route_bias_keyed_residual_ratio_matched", "sare_phase_memory_route_bias_keyed_residual_per_expert_scale", "sare_phase_memory_route_bias_keyed_residual_learned_scale", "sare_phase_memory_route_bias_keyed_residual_topk_masked", "sare_phase_memory_route_bias_keyed_residual_postact", "sare_phase_memory_route_bias_keyed_residual_centered", "sare_phase_memory_route_bias_keyed_residual_boosted", "sare_phase_memory_route_bias_keyed_residual_entropy_gate", "sare_phase_memory_route_bias_contextual", "sare_phase_memory_route_bias_scale_gate", "sare_phase_memory_route_bias_normalized", "sare_phase_memory_route_bias_layernorm", "sare_phase_memory_route_bias_gated", "sare_phase_memory_route_bias_dual", "treg_h", "srw", "por"]
    for variant in variants:
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
    variants = ["flat_dense", "token_dense", "token_gru", "single_expert", "sare", "sare_phase_memory", "sare_phase_memory_gated", "sare_phase_memory_route_bias", "sare_phase_memory_route_bias_keyed", "sare_phase_memory_route_bias_keyed_residual", "sare_phase_memory_route_bias_keyed_residual_predictive", "sare_phase_memory_route_bias_keyed_residual_predictive_blend", "sare_phase_memory_route_bias_keyed_residual_predictive_capped", "sare_phase_memory_route_bias_keyed_residual_predictive_delta_mix", "sare_phase_memory_route_bias_keyed_residual_predictive_positive", "sare_phase_memory_route_bias_keyed_residual_predictive_aligned", "sare_phase_memory_route_bias_keyed_residual_predictive_weak_prior_replace", "sare_phase_memory_route_bias_keyed_residual_predictive_weak_prior_assist", "sare_phase_memory_route_bias_keyed_residual_predictive_weak_prior_top1_assist", "sare_phase_memory_route_bias_keyed_residual_predictive_weak_prior_top1_max_merge", "sare_phase_memory_route_bias_keyed_residual_predictive_top1_consensus_bonus", "sare_phase_memory_route_bias_keyed_residual_predictive_delta_gate", "sare_phase_memory_route_bias_keyed_residual_ema", "sare_phase_memory_route_bias_keyed_residual_margin_gate", "sare_phase_memory_route_bias_keyed_residual_mild_boost", "sare_phase_memory_route_bias_keyed_residual_challenger", "sare_phase_memory_route_bias_keyed_residual_base_clip", "sare_phase_memory_route_bias_keyed_residual_ratio_gate", "sare_phase_memory_route_bias_keyed_residual_orthogonal", "sare_phase_memory_route_bias_keyed_residual_floor_boost", "sare_phase_memory_route_bias_keyed_residual_target_norm", "sare_phase_memory_route_bias_keyed_residual_ratio_matched", "sare_phase_memory_route_bias_keyed_residual_per_expert_scale", "sare_phase_memory_route_bias_keyed_residual_learned_scale", "sare_phase_memory_route_bias_keyed_residual_topk_masked", "sare_phase_memory_route_bias_keyed_residual_postact", "sare_phase_memory_route_bias_keyed_residual_centered", "sare_phase_memory_route_bias_keyed_residual_boosted", "sare_phase_memory_route_bias_keyed_residual_entropy_gate", "sare_phase_memory_route_bias_contextual", "sare_phase_memory_route_bias_scale_gate", "sare_phase_memory_route_bias_normalized", "sare_phase_memory_route_bias_layernorm", "sare_phase_memory_route_bias_gated", "sare_phase_memory_route_bias_dual", "treg_h", "srw", "por"]
    for variant in variants:
        model_config = ModelConfig(variant=variant, hidden_size=64, token_dim=64, patch_size=8)
        model = build_model(model_config, observation_space, action_space)
        state = model.initial_state(batch_size=1, device=torch.device("cpu"))
        output = model.forward(obs, state=state, done=done)
        assert output.logits.shape == (1, action_space.n)
        assert output.value.shape == (1,)
