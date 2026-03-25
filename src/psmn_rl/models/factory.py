from __future__ import annotations

import gymnasium as gym

from psmn_rl.config import ModelConfig
from psmn_rl.models.common import ActorCriticModel
from psmn_rl.models.cores.dense import FlatDenseCore, TokenDenseCore
from psmn_rl.models.cores.recurrent import TokenGRUCore
from psmn_rl.models.options.por import PORCore
from psmn_rl.models.relational.srw import SRWCore
from psmn_rl.models.routing.sare import (
    RoutedExpertCore,
    RoutedExpertGatedPhaseMemoryCore,
    RoutedExpertPhaseMemoryCore,
    RoutedExpertRouteBiasedContextualPhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualCenteredPhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualPredictiveDeltaMixPhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualEmaPhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualPredictiveCappedPhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualPredictivePhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualPredictiveBlendPhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualEntropyGatePhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualMarginGatePhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualBoostedPhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualMildBoostPhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualBaseClippedPhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualChallengerPhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualRatioGatedPhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualFloorBoostPhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualOrthogonalPhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualRatioMatchedPhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualPerExpertScalePhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualLearnedScalePhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualPhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualTargetNormPhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualTopKMaskedPhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedResidualPostActivationPhaseMemoryCore,
    RoutedExpertRouteBiasedKeyedPhaseMemoryCore,
    RoutedExpertRouteBiasedLayerNormPhaseMemoryCore,
    RoutedExpertRouteBiasedNormalizedPhaseMemoryCore,
    RoutedExpertRouteBiasedPhaseMemoryCore,
    RoutedExpertRouteBiasedScaleGatedPhaseMemoryCore,
    RoutedExpertRouteBiasedDualPhaseMemoryCore,
    RoutedExpertRouteBiasedGatedPhaseMemoryCore,
)
from psmn_rl.models.routing.treg_h import TREGHCore


def build_model(model_config: ModelConfig, observation_space: gym.Space, action_space: gym.Space) -> ActorCriticModel:
    if not hasattr(action_space, "n"):
        raise ValueError("Only discrete action spaces are currently supported")
    variant = model_config.variant
    if variant == "flat_dense":
        core = FlatDenseCore(observation_space, hidden_size=model_config.hidden_size)
    elif variant == "token_dense":
        core = TokenDenseCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            num_heads=model_config.num_heads,
            num_layers=model_config.encoder_layers,
            dropout=model_config.dropout,
        )
    elif variant == "token_gru":
        core = TokenGRUCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            num_heads=model_config.num_heads,
            num_layers=model_config.encoder_layers,
            dropout=model_config.dropout,
        )
    elif variant == "single_expert":
        core = RoutedExpertCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=1,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=1,
            temperature=model_config.temperature,
        )
    elif variant == "sare":
        core = RoutedExpertCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
        )
    elif variant == "sare_phase_memory":
        core = RoutedExpertPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
        )
    elif variant == "sare_phase_memory_gated":
        core = RoutedExpertGatedPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            memory_gate_bias=model_config.memory_gate_bias,
            memory_reset_bias=model_config.memory_reset_bias,
        )
    elif variant == "sare_phase_memory_route_bias":
        core = RoutedExpertRouteBiasedPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed":
        core = RoutedExpertRouteBiasedKeyedPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual":
        core = RoutedExpertRouteBiasedKeyedResidualPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual_predictive":
        core = RoutedExpertRouteBiasedKeyedResidualPredictivePhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual_predictive_blend":
        core = RoutedExpertRouteBiasedKeyedResidualPredictiveBlendPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual_predictive_capped":
        core = RoutedExpertRouteBiasedKeyedResidualPredictiveCappedPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual_predictive_delta_mix":
        core = RoutedExpertRouteBiasedKeyedResidualPredictiveDeltaMixPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual_ema":
        core = RoutedExpertRouteBiasedKeyedResidualEmaPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual_margin_gate":
        core = RoutedExpertRouteBiasedKeyedResidualMarginGatePhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual_mild_boost":
        core = RoutedExpertRouteBiasedKeyedResidualMildBoostPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual_challenger":
        core = RoutedExpertRouteBiasedKeyedResidualChallengerPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual_base_clip":
        core = RoutedExpertRouteBiasedKeyedResidualBaseClippedPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual_ratio_gate":
        core = RoutedExpertRouteBiasedKeyedResidualRatioGatedPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual_orthogonal":
        core = RoutedExpertRouteBiasedKeyedResidualOrthogonalPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual_floor_boost":
        core = RoutedExpertRouteBiasedKeyedResidualFloorBoostPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual_target_norm":
        core = RoutedExpertRouteBiasedKeyedResidualTargetNormPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual_ratio_matched":
        core = RoutedExpertRouteBiasedKeyedResidualRatioMatchedPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual_per_expert_scale":
        core = RoutedExpertRouteBiasedKeyedResidualPerExpertScalePhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual_learned_scale":
        core = RoutedExpertRouteBiasedKeyedResidualLearnedScalePhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual_topk_masked":
        core = RoutedExpertRouteBiasedKeyedResidualTopKMaskedPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual_postact":
        core = RoutedExpertRouteBiasedKeyedResidualPostActivationPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual_centered":
        core = RoutedExpertRouteBiasedKeyedResidualCenteredPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual_boosted":
        core = RoutedExpertRouteBiasedKeyedResidualBoostedPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_keyed_residual_entropy_gate":
        core = RoutedExpertRouteBiasedKeyedResidualEntropyGatePhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_contextual":
        core = RoutedExpertRouteBiasedContextualPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_scale_gate":
        core = RoutedExpertRouteBiasedScaleGatedPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_normalized":
        core = RoutedExpertRouteBiasedNormalizedPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_layernorm":
        core = RoutedExpertRouteBiasedLayerNormPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_gated":
        core = RoutedExpertRouteBiasedGatedPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            memory_gate_bias=model_config.memory_gate_bias,
            memory_reset_bias=model_config.memory_reset_bias,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "sare_phase_memory_route_bias_dual":
        core = RoutedExpertRouteBiasedDualPhaseMemoryCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            memory_mix=model_config.memory_mix,
            route_memory_scale=model_config.route_memory_scale,
        )
    elif variant == "treg_h":
        core = TREGHCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            temperature=model_config.temperature,
            ponder_cost=model_config.ponder_cost,
        )
    elif variant == "srw":
        core = SRWCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            expert_count=model_config.expert_count,
            expert_hidden_size=model_config.expert_hidden_size,
            top_k=model_config.top_k,
            temperature=model_config.temperature,
            num_heads=model_config.num_heads,
            relational_tokens=model_config.relational_tokens,
        )
    elif variant == "por":
        core = PORCore(
            observation_space=observation_space,
            token_dim=model_config.token_dim,
            patch_size=model_config.patch_size,
            hidden_size=model_config.hidden_size,
            num_heads=model_config.num_heads,
            num_layers=model_config.encoder_layers,
            dropout=model_config.dropout,
            option_count=model_config.option_count,
        )
    else:
        raise ValueError(f"Unknown model variant: {variant}")
    return ActorCriticModel(core=core, action_dim=action_space.n, hidden_size=model_config.hidden_size)
