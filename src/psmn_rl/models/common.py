from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


TensorDict = dict[str, torch.Tensor]


class _ParameterCollection:
    def __init__(self, *modules: nn.Module) -> None:
        self._modules = modules

    def parameters(self, recurse: bool = True):
        seen: set[int] = set()
        for module in self._modules:
            for parameter in module.parameters(recurse=recurse):
                parameter_id = id(parameter)
                if parameter_id in seen:
                    continue
                seen.add(parameter_id)
                yield parameter


@dataclass
class CoreOutput:
    pooled: torch.Tensor
    tokens: torch.Tensor | None
    metrics: dict[str, torch.Tensor | float] = field(default_factory=dict)
    next_state: TensorDict = field(default_factory=dict)
    aux_losses: dict[str, torch.Tensor] = field(default_factory=dict)
    logit_bias: torch.Tensor | None = None
    policy_features: TensorDict = field(default_factory=dict)


@dataclass
class ModelOutput:
    logits: torch.Tensor
    value: torch.Tensor
    metrics: dict[str, torch.Tensor | float]
    next_state: TensorDict
    aux_losses: dict[str, torch.Tensor]


def masked_mean(tokens: torch.Tensor) -> torch.Tensor:
    return tokens.mean(dim=1)


def entropy_from_probs(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=dim)


def pooled_representation_metrics(pooled: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"repr/pooled_norm": pooled.norm(dim=-1)}


def token_representation_metrics(tokens: torch.Tensor, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
    spatial_tokens = tokens[:, :-1] if tokens.size(1) > 1 else tokens
    direction_token = tokens[:, -1]
    metrics: dict[str, torch.Tensor] = {
        "repr/pooled_norm": pooled.norm(dim=-1),
        "repr/token_norm": spatial_tokens.norm(dim=-1).mean(dim=-1),
        "repr/token_feature_std": spatial_tokens.std(dim=1, unbiased=False).mean(dim=-1),
        "repr/direction_norm": direction_token.norm(dim=-1),
    }
    if spatial_tokens.size(1) > 1:
        normalized = F.normalize(spatial_tokens, dim=-1)
        cosine = normalized @ normalized.transpose(1, 2)
        token_count = spatial_tokens.size(1)
        off_diag = (cosine.sum(dim=(1, 2)) - token_count) / max(token_count * (token_count - 1), 1)
        metrics["repr/token_pairwise_cosine"] = off_diag
    else:
        metrics["repr/token_pairwise_cosine"] = torch.zeros(
            spatial_tokens.size(0), device=tokens.device, dtype=tokens.dtype
        )
    return metrics


class ActorCriticModel(nn.Module):
    def __init__(
        self,
        core: nn.Module,
        action_dim: int,
        hidden_size: int,
        *,
        policy_margin_residual: bool = False,
        policy_margin_residual_scale: float = 1.0,
        policy_margin_threshold: float = 0.25,
        policy_margin_sharpness: float = 12.0,
        policy_logit_gain: bool = False,
        policy_logit_gain_scale: float = 0.5,
        policy_logit_gain_threshold: float = 0.25,
        policy_logit_gain_sharpness: float = 12.0,
        policy_top2_rerank: bool = False,
        policy_top2_rerank_scale: float = 0.5,
        policy_top2_rerank_threshold: float = 0.25,
        policy_top2_rerank_sharpness: float = 12.0,
        policy_option_margin_adapter: bool = False,
        policy_option_margin_adapter_scale: float = 0.5,
        policy_option_margin_threshold: float = 0.25,
        policy_option_margin_sharpness: float = 12.0,
        policy_option_top2_rerank: bool = False,
        policy_option_top2_rerank_scale: float = 0.5,
        policy_option_top2_rerank_threshold: float = 0.25,
        policy_option_top2_rerank_sharpness: float = 12.0,
        policy_option_top2_use_duration_gate: bool = False,
        policy_option_hidden_film: bool = False,
        policy_option_hidden_film_scale: float = 0.5,
        policy_option_hidden_use_duration_gate: bool = False,
        policy_option_hidden_duration_mix: float = 1.0,
        policy_option_hidden_branch_gates: bool = False,
        policy_option_hidden_scale_duration_mix: float = 1.0,
        policy_option_hidden_shift_duration_mix: float = 1.0,
        policy_option_hidden_low_rank: bool = False,
        policy_option_hidden_low_rank_dim: int = 32,
        policy_option_hidden_split_heads: bool = False,
        policy_option_hidden_scale_only: bool = False,
        policy_option_hidden_scale_weight: float = 1.0,
        policy_option_hidden_adaptive_scale_floor: bool = False,
        policy_option_hidden_scale_floor: float = 0.0,
        policy_option_hidden_scale_floor_power: float = 1.0,
        policy_option_hidden_gate_bias: bool = False,
        policy_option_hidden_gate_bias_scale: float = 0.0,
        policy_option_hidden_scale_gate_bias_scale: float = 0.0,
        policy_option_hidden_shift_gate_bias_scale: float = 0.0,
        policy_option_hidden_adaptive_shift_floor: bool = False,
        policy_option_hidden_shift_floor: float = 0.0,
        policy_option_hidden_shift_floor_power: float = 1.0,
        policy_option_hidden_shift_compensation: bool = False,
        policy_option_hidden_shift_compensation_scale: float = 0.0,
        policy_option_hidden_low_margin_gate: bool = False,
        policy_option_hidden_margin_threshold: float = 0.25,
        policy_option_hidden_margin_sharpness: float = 12.0,
        policy_option_hidden_blend_gate: bool = False,
        policy_option_hidden_blend_scale: float = 1.0,
        policy_option_hidden_shift_weight: float = 1.0,
        policy_option_hidden_center_shift: bool = False,
        policy_option_hidden_center_shift_scale: float = 1.0,
        policy_option_hidden_bound_shift: bool = False,
        policy_option_hidden_shift_bound_scale: float = 1.0,
        policy_option_hidden_post_norm: bool = False,
    ) -> None:
        super().__init__()
        self.core = core
        self.policy_margin_residual = policy_margin_residual
        self.policy_margin_residual_scale = policy_margin_residual_scale
        self.policy_margin_threshold = policy_margin_threshold
        self.policy_margin_sharpness = policy_margin_sharpness
        self.policy_logit_gain = policy_logit_gain
        self.policy_logit_gain_scale = policy_logit_gain_scale
        self.policy_logit_gain_threshold = policy_logit_gain_threshold
        self.policy_logit_gain_sharpness = policy_logit_gain_sharpness
        self.policy_top2_rerank = policy_top2_rerank
        self.policy_top2_rerank_scale = policy_top2_rerank_scale
        self.policy_top2_rerank_threshold = policy_top2_rerank_threshold
        self.policy_top2_rerank_sharpness = policy_top2_rerank_sharpness
        self.policy_option_margin_adapter = policy_option_margin_adapter
        self.policy_option_margin_adapter_scale = policy_option_margin_adapter_scale
        self.policy_option_margin_threshold = policy_option_margin_threshold
        self.policy_option_margin_sharpness = policy_option_margin_sharpness
        self.policy_option_top2_rerank = policy_option_top2_rerank
        self.policy_option_top2_rerank_scale = policy_option_top2_rerank_scale
        self.policy_option_top2_rerank_threshold = policy_option_top2_rerank_threshold
        self.policy_option_top2_rerank_sharpness = policy_option_top2_rerank_sharpness
        self.policy_option_top2_use_duration_gate = policy_option_top2_use_duration_gate
        self.policy_option_hidden_film = policy_option_hidden_film
        self.policy_option_hidden_film_scale = policy_option_hidden_film_scale
        self.policy_option_hidden_use_duration_gate = policy_option_hidden_use_duration_gate
        self.policy_option_hidden_duration_mix = max(0.0, min(1.0, policy_option_hidden_duration_mix))
        self.policy_option_hidden_branch_gates = policy_option_hidden_branch_gates
        self.policy_option_hidden_scale_duration_mix = max(0.0, min(1.0, policy_option_hidden_scale_duration_mix))
        self.policy_option_hidden_shift_duration_mix = max(0.0, min(1.0, policy_option_hidden_shift_duration_mix))
        self.policy_option_hidden_low_rank = policy_option_hidden_low_rank
        self.policy_option_hidden_low_rank_dim = max(1, int(policy_option_hidden_low_rank_dim))
        self.policy_option_hidden_split_heads = policy_option_hidden_split_heads
        self.policy_option_hidden_scale_only = policy_option_hidden_scale_only
        self.policy_option_hidden_scale_weight = max(0.0, policy_option_hidden_scale_weight)
        self.policy_option_hidden_adaptive_scale_floor = policy_option_hidden_adaptive_scale_floor
        self.policy_option_hidden_scale_floor = max(0.0, policy_option_hidden_scale_floor)
        self.policy_option_hidden_scale_floor_power = max(1e-6, policy_option_hidden_scale_floor_power)
        self.policy_option_hidden_gate_bias_scale = max(0.0, policy_option_hidden_gate_bias_scale)
        self.policy_option_hidden_scale_gate_bias_scale = max(0.0, policy_option_hidden_scale_gate_bias_scale)
        self.policy_option_hidden_shift_gate_bias_scale = max(0.0, policy_option_hidden_shift_gate_bias_scale)
        self.policy_option_hidden_gate_bias = (
            policy_option_hidden_gate_bias
            or self.policy_option_hidden_gate_bias_scale > 0.0
            or self.policy_option_hidden_scale_gate_bias_scale > 0.0
            or self.policy_option_hidden_shift_gate_bias_scale > 0.0
        )
        self.policy_option_hidden_adaptive_shift_floor = policy_option_hidden_adaptive_shift_floor
        self.policy_option_hidden_shift_floor = max(0.0, policy_option_hidden_shift_floor)
        self.policy_option_hidden_shift_floor_power = max(1e-6, policy_option_hidden_shift_floor_power)
        self.policy_option_hidden_shift_compensation = policy_option_hidden_shift_compensation
        self.policy_option_hidden_shift_compensation_scale = max(0.0, policy_option_hidden_shift_compensation_scale)
        self.policy_option_hidden_low_margin_gate = policy_option_hidden_low_margin_gate
        self.policy_option_hidden_margin_threshold = policy_option_hidden_margin_threshold
        self.policy_option_hidden_margin_sharpness = policy_option_hidden_margin_sharpness
        self.policy_option_hidden_blend_gate = policy_option_hidden_blend_gate
        self.policy_option_hidden_blend_scale = max(0.0, policy_option_hidden_blend_scale)
        self.policy_option_hidden_shift_weight = max(0.0, policy_option_hidden_shift_weight)
        self.policy_option_hidden_center_shift = policy_option_hidden_center_shift
        self.policy_option_hidden_center_shift_scale = max(0.0, policy_option_hidden_center_shift_scale)
        self.policy_option_hidden_bound_shift = policy_option_hidden_bound_shift
        self.policy_option_hidden_shift_bound_scale = max(0.0, policy_option_hidden_shift_bound_scale)
        self.policy_option_hidden_post_norm = policy_option_hidden_post_norm
        self.policy_norm = nn.LayerNorm(hidden_size)
        self.policy_hidden = nn.Linear(hidden_size, hidden_size)
        self.policy_activation = nn.GELU()
        self.policy_out = nn.Linear(hidden_size, action_dim)
        self._policy_head_view = _ParameterCollection(self.policy_norm, self.policy_hidden, self.policy_out)
        if self.policy_top2_rerank:
            top2_embed_dim = min(hidden_size, 32)
            self.policy_top2_action_embed = nn.Embedding(action_dim, top2_embed_dim)
            self.policy_top2_head = nn.Sequential(
                nn.LayerNorm(hidden_size + top2_embed_dim * 2 + 2),
                nn.Linear(hidden_size + top2_embed_dim * 2 + 2, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )
            self.policy_top2_gate = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )
        if self.policy_margin_residual:
            self.policy_residual_head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, action_dim),
            )
            self.policy_residual_gate = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )
        if self.policy_logit_gain:
            self.policy_gain_head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, action_dim),
            )
            self.policy_gain_gate = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )
        if self.policy_option_margin_adapter:
            self.policy_option_margin_gate = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )
            self.policy_option_margin_adapter_head = nn.Sequential(
                nn.LazyLinear(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, action_dim),
            )
            final_linear = self.policy_option_margin_adapter_head[-1]
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)
        if self.policy_option_top2_rerank:
            option_top2_embed_dim = min(hidden_size, 32)
            self.policy_option_top2_action_embed = nn.Embedding(action_dim, option_top2_embed_dim)
            self.policy_option_top2_head = nn.Sequential(
                nn.LazyLinear(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )
        if self.policy_option_hidden_film:
            hidden_film_width = self.policy_option_hidden_low_rank_dim if self.policy_option_hidden_low_rank else hidden_size
            if self.policy_option_hidden_split_heads:
                self.policy_option_hidden_scale_head = nn.Sequential(
                    nn.LazyLinear(hidden_film_width),
                    nn.GELU(),
                    nn.Linear(hidden_film_width, hidden_size),
                )
                self.policy_option_hidden_shift_head = nn.Sequential(
                    nn.LazyLinear(hidden_film_width),
                    nn.GELU(),
                    nn.Linear(hidden_film_width, hidden_size),
                )
                final_scale_linear = self.policy_option_hidden_scale_head[-1]
                final_shift_linear = self.policy_option_hidden_shift_head[-1]
                nn.init.zeros_(final_scale_linear.weight)
                nn.init.zeros_(final_scale_linear.bias)
                nn.init.zeros_(final_shift_linear.weight)
                nn.init.zeros_(final_shift_linear.bias)
            else:
                self.policy_option_hidden_film_head = nn.Sequential(
                    nn.LazyLinear(hidden_film_width),
                    nn.GELU(),
                    nn.Linear(hidden_film_width, hidden_size * 2),
                )
                final_linear = self.policy_option_hidden_film_head[-1]
                nn.init.zeros_(final_linear.weight)
                nn.init.zeros_(final_linear.bias)
            if self.policy_option_hidden_blend_gate:
                self.policy_option_hidden_blend_head = nn.Sequential(
                    nn.LazyLinear(hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, 1),
                )
                final_blend_linear = self.policy_option_hidden_blend_head[-1]
                nn.init.zeros_(final_blend_linear.weight)
                nn.init.constant_(final_blend_linear.bias, 2.0)
            if self.policy_option_hidden_gate_bias:
                self.policy_option_hidden_gate_bias_head = nn.Sequential(
                    nn.LazyLinear(hidden_film_width),
                    nn.GELU(),
                    nn.Linear(hidden_film_width, 1),
                )
                final_gate_bias_linear = self.policy_option_hidden_gate_bias_head[-1]
                nn.init.zeros_(final_gate_bias_linear.weight)
                nn.init.zeros_(final_gate_bias_linear.bias)
            if self.policy_option_hidden_post_norm:
                self.policy_option_hidden_post_layernorm = nn.LayerNorm(hidden_size)
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    @property
    def policy_head(self) -> _ParameterCollection:
        return self._policy_head_view

    def initial_state(self, batch_size: int, device: torch.device) -> TensorDict:
        if hasattr(self.core, "initial_state"):
            return getattr(self.core, "initial_state")(batch_size, device)
        return {}

    def forward(self, obs: TensorDict, state: TensorDict | None = None, done: torch.Tensor | None = None) -> ModelOutput:
        core_output: CoreOutput = self.core(obs, state or {}, done)
        metrics = dict(core_output.metrics)
        policy_hidden = self.policy_activation(self.policy_hidden(self.policy_norm(core_output.pooled)))
        if self.policy_option_hidden_film:
            option_actor_features = core_output.policy_features.get("option_actor_features")
            option_actor_stability = core_output.policy_features.get("option_actor_stability")
            option_actor_duration_gate = core_output.policy_features.get("option_actor_duration_gate")
            if option_actor_features is not None and option_actor_stability is not None:
                base_policy_hidden = policy_hidden
                low_margin_gate = None
                margin_before = None
                if self.policy_option_hidden_low_margin_gate:
                    with torch.no_grad():
                        pre_film_logits = self.policy_out(base_policy_hidden)
                        top2 = torch.topk(pre_film_logits, k=min(2, pre_film_logits.size(-1)), dim=-1).values
                        if top2.size(-1) > 1:
                            margin_before = top2[..., 0] - top2[..., 1]
                        else:
                            margin_before = top2[..., 0]
                        low_margin_gate = torch.sigmoid(
                            (self.policy_option_hidden_margin_threshold - margin_before)
                            * self.policy_option_hidden_margin_sharpness
                        ).to(policy_hidden.dtype)
                if self.policy_option_hidden_use_duration_gate and option_actor_duration_gate is not None:
                    mix = self.policy_option_hidden_duration_mix
                    option_gate_signal = (mix * option_actor_duration_gate) + ((1.0 - mix) * option_actor_stability)
                    if self.policy_option_hidden_branch_gates:
                        scale_gate_signal = (
                            self.policy_option_hidden_scale_duration_mix * option_actor_duration_gate
                        ) + ((1.0 - self.policy_option_hidden_scale_duration_mix) * option_actor_stability)
                        shift_gate_signal = (
                            self.policy_option_hidden_shift_duration_mix * option_actor_duration_gate
                        ) + ((1.0 - self.policy_option_hidden_shift_duration_mix) * option_actor_stability)
                    else:
                        scale_gate_signal = option_gate_signal
                        shift_gate_signal = option_gate_signal
                else:
                    option_gate_signal = option_actor_stability
                    scale_gate_signal = option_gate_signal
                    shift_gate_signal = option_gate_signal
                gate_bias = None
                if self.policy_option_hidden_split_heads:
                    raw_scale = self.policy_option_hidden_scale_head(option_actor_features)
                    raw_shift = self.policy_option_hidden_shift_head(option_actor_features)
                else:
                    raw_film = self.policy_option_hidden_film_head(option_actor_features)
                    raw_scale, raw_shift = raw_film.chunk(2, dim=-1)
                if self.policy_option_hidden_gate_bias:
                    gate_bias = torch.tanh(self.policy_option_hidden_gate_bias_head(option_actor_features))
                    scale_gate_bias = gate_bias * (
                        self.policy_option_hidden_gate_bias_scale
                        + self.policy_option_hidden_scale_gate_bias_scale
                    )
                    shift_gate_bias = gate_bias * (
                        self.policy_option_hidden_gate_bias_scale
                        + self.policy_option_hidden_shift_gate_bias_scale
                    )
                    scale_gate_signal = (scale_gate_signal + scale_gate_bias).clamp(0.0, 1.0)
                    shift_gate_signal = (shift_gate_signal + shift_gate_bias).clamp(0.0, 1.0)
                adaptive_scale = self.policy_option_hidden_film_scale
                if self.policy_option_hidden_adaptive_scale_floor:
                    floor_gate_signal = scale_gate_signal.pow(self.policy_option_hidden_scale_floor_power)
                    adaptive_scale = (
                        self.policy_option_hidden_scale_floor
                        + (self.policy_option_hidden_film_scale - self.policy_option_hidden_scale_floor)
                        * floor_gate_signal
                    )
                adaptive_shift_weight = shift_gate_signal.new_full(
                    shift_gate_signal.shape,
                    self.policy_option_hidden_shift_weight,
                )
                if self.policy_option_hidden_adaptive_shift_floor:
                    shift_floor_gate_signal = shift_gate_signal.pow(self.policy_option_hidden_shift_floor_power)
                    adaptive_shift_weight = (
                        self.policy_option_hidden_shift_floor
                        + (self.policy_option_hidden_shift_weight - self.policy_option_hidden_shift_floor)
                        * shift_floor_gate_signal
                    )
                shift_compensation = 1.0
                if self.policy_option_hidden_shift_compensation and self.policy_option_hidden_film_scale > 0.0:
                    shift_compensation = 1.0 + (
                        self.policy_option_hidden_shift_compensation_scale
                        * (self.policy_option_hidden_film_scale - adaptive_scale)
                        / self.policy_option_hidden_film_scale
                    )
                scale_gate = (adaptive_scale * scale_gate_signal).to(policy_hidden.dtype)
                shift_gate = (
                    self.policy_option_hidden_film_scale * shift_gate_signal * shift_compensation
                ).to(policy_hidden.dtype)
                if low_margin_gate is not None:
                    scale_gate = scale_gate * low_margin_gate.unsqueeze(-1)
                    shift_gate = shift_gate * low_margin_gate.unsqueeze(-1)
                film_scale = torch.tanh(raw_scale) * scale_gate * self.policy_option_hidden_scale_weight
                if self.policy_option_hidden_scale_only:
                    film_shift = torch.zeros_like(policy_hidden)
                else:
                    if self.policy_option_hidden_bound_shift:
                        film_shift = (
                            torch.tanh(raw_shift)
                            * shift_gate
                            * adaptive_shift_weight
                            * self.policy_option_hidden_shift_bound_scale
                        )
                    else:
                        film_shift = raw_shift * shift_gate * adaptive_shift_weight
                    if self.policy_option_hidden_center_shift:
                        film_shift = film_shift - film_shift.mean(dim=-1, keepdim=True)
                        film_shift = film_shift * self.policy_option_hidden_center_shift_scale
                policy_hidden = policy_hidden * (1.0 + film_scale) + film_shift
                blend_gate = None
                if self.policy_option_hidden_blend_gate:
                    blend_gate = torch.sigmoid(self.policy_option_hidden_blend_head(option_actor_features)).squeeze(-1)
                    blend_gate = (blend_gate * self.policy_option_hidden_blend_scale).clamp(0.0, 1.0)
                    policy_hidden = base_policy_hidden + blend_gate.unsqueeze(-1) * (policy_hidden - base_policy_hidden)
                if self.policy_option_hidden_post_norm:
                    policy_hidden = self.policy_option_hidden_post_layernorm(policy_hidden)
                metrics.update(
                    {
                        "policy/option_hidden_film_gate_signal_mean": option_gate_signal.squeeze(-1),
                        "policy/option_hidden_film_gate_mean": 0.5
                        * (scale_gate.mean(dim=-1) + shift_gate.mean(dim=-1)),
                        "policy/option_hidden_film_stability_mean": option_actor_stability.squeeze(-1),
                        "policy/option_hidden_film_duration_mix": float(self.policy_option_hidden_duration_mix),
                        "policy/option_hidden_branch_gates": float(self.policy_option_hidden_branch_gates),
                        "policy/option_hidden_scale_duration_mix": float(self.policy_option_hidden_scale_duration_mix),
                        "policy/option_hidden_shift_duration_mix": float(self.policy_option_hidden_shift_duration_mix),
                        "policy/option_hidden_low_rank": float(self.policy_option_hidden_low_rank),
                        "policy/option_hidden_low_rank_dim": float(self.policy_option_hidden_low_rank_dim),
                        "policy/option_hidden_scale_gate_signal_mean": scale_gate_signal.squeeze(-1),
                        "policy/option_hidden_shift_gate_signal_mean": shift_gate_signal.squeeze(-1),
                        "policy/option_hidden_scale_gate_mean": scale_gate.mean(dim=-1),
                        "policy/option_hidden_shift_gate_mean": shift_gate.mean(dim=-1),
                        "policy/option_hidden_split_heads": float(self.policy_option_hidden_split_heads),
                        "policy/option_hidden_film_scale_only": float(self.policy_option_hidden_scale_only),
                        "policy/option_hidden_film_scale_weight": float(self.policy_option_hidden_scale_weight),
                        "policy/option_hidden_adaptive_scale_floor": float(self.policy_option_hidden_adaptive_scale_floor),
                        "policy/option_hidden_scale_floor": float(self.policy_option_hidden_scale_floor),
                        "policy/option_hidden_scale_floor_power": float(self.policy_option_hidden_scale_floor_power),
                        "policy/option_hidden_gate_bias": float(self.policy_option_hidden_gate_bias),
                        "policy/option_hidden_gate_bias_scale": float(self.policy_option_hidden_gate_bias_scale),
                        "policy/option_hidden_scale_gate_bias_scale": float(
                            self.policy_option_hidden_scale_gate_bias_scale
                        ),
                        "policy/option_hidden_shift_gate_bias_scale": float(
                            self.policy_option_hidden_shift_gate_bias_scale
                        ),
                        "policy/option_hidden_adaptive_shift_floor": float(
                            self.policy_option_hidden_adaptive_shift_floor
                        ),
                        "policy/option_hidden_shift_floor": float(self.policy_option_hidden_shift_floor),
                        "policy/option_hidden_shift_floor_power": float(self.policy_option_hidden_shift_floor_power),
                        "policy/option_hidden_shift_compensation": float(self.policy_option_hidden_shift_compensation),
                        "policy/option_hidden_shift_compensation_scale": float(
                            self.policy_option_hidden_shift_compensation_scale
                        ),
                        "policy/option_hidden_film_low_margin_gate": float(self.policy_option_hidden_low_margin_gate),
                        "policy/option_hidden_film_margin_threshold": float(self.policy_option_hidden_margin_threshold),
                        "policy/option_hidden_blend_gate": float(self.policy_option_hidden_blend_gate),
                        "policy/option_hidden_blend_scale": float(self.policy_option_hidden_blend_scale),
                        "policy/option_hidden_film_shift_weight": float(self.policy_option_hidden_shift_weight),
                        "policy/option_hidden_center_shift": float(self.policy_option_hidden_center_shift),
                        "policy/option_hidden_center_shift_scale": float(self.policy_option_hidden_center_shift_scale),
                        "policy/option_hidden_bound_shift": float(self.policy_option_hidden_bound_shift),
                        "policy/option_hidden_shift_bound_scale": float(self.policy_option_hidden_shift_bound_scale),
                        "policy/option_hidden_post_norm": float(self.policy_option_hidden_post_norm),
                        "policy/option_hidden_adaptive_scale_mean": scale_gate.mean(dim=-1),
                        "policy/option_hidden_adaptive_shift_weight_mean": adaptive_shift_weight.squeeze(-1),
                        "policy/option_hidden_shift_compensation_mean": shift_gate.mean(dim=-1),
                        "policy/option_hidden_film_scale_norm": film_scale.norm(dim=-1),
                        "policy/option_hidden_film_shift_norm": film_shift.norm(dim=-1),
                        "policy/option_hidden_center_shift_mean_abs": film_shift.mean(dim=-1).abs(),
                        "policy/option_hidden_post_hidden_norm": policy_hidden.norm(dim=-1),
                    }
                )
                if blend_gate is not None:
                    metrics["policy/option_hidden_blend_gate_mean"] = blend_gate
                if gate_bias is not None:
                    metrics["policy/option_hidden_gate_bias_mean"] = gate_bias.squeeze(-1)
                    metrics["policy/option_hidden_scale_gate_bias_mean"] = scale_gate_bias.squeeze(-1)
                    metrics["policy/option_hidden_shift_gate_bias_mean"] = shift_gate_bias.squeeze(-1)
                if low_margin_gate is not None and margin_before is not None:
                    metrics["policy/option_hidden_film_low_margin_gate_mean"] = low_margin_gate
                    metrics["policy/option_hidden_film_margin_before"] = margin_before
        base_logits = self.policy_out(policy_hidden)
        logits = base_logits
        if core_output.logit_bias is not None:
            logits = logits + core_output.logit_bias
        if self.policy_top2_rerank and logits.size(-1) > 1:
            top2 = torch.topk(logits, k=2, dim=-1)
            top2_values = top2.values
            top2_indices = top2.indices
            margin = top2_values[..., 0] - top2_values[..., 1]
            low_margin_gate = torch.sigmoid(
                (self.policy_top2_rerank_threshold - margin) * self.policy_top2_rerank_sharpness
            )
            learned_gate = torch.sigmoid(self.policy_top2_gate(core_output.pooled)).squeeze(-1)
            gate = low_margin_gate * learned_gate
            top1_embed = self.policy_top2_action_embed(top2_indices[..., 0])
            top2_embed = self.policy_top2_action_embed(top2_indices[..., 1])
            rerank_features = torch.cat(
                [
                    core_output.pooled,
                    top1_embed,
                    top2_embed,
                    top2_values,
                ],
                dim=-1,
            )
            raw_delta = torch.tanh(self.policy_top2_head(rerank_features)).squeeze(-1) * self.policy_top2_rerank_scale
            effective_delta = gate * raw_delta
            effective_delta = effective_delta.to(logits.dtype)
            correction = torch.zeros_like(logits)
            correction.scatter_add_(1, top2_indices[..., 0].unsqueeze(-1), effective_delta.unsqueeze(-1))
            correction.scatter_add_(1, top2_indices[..., 1].unsqueeze(-1), (-effective_delta).unsqueeze(-1))
            logits = logits + correction
            top2_after = torch.topk(logits, k=2, dim=-1).values
            metrics.update(
                {
                    "policy/top2_rerank_gate_mean": gate,
                    "policy/top2_rerank_gate_max": gate.max(),
                    "policy/top2_rerank_low_margin_mean": low_margin_gate,
                    "policy/top2_rerank_margin_before": margin,
                    "policy/top2_rerank_margin_after": top2_after[..., 0] - top2_after[..., 1],
                    "policy/top2_rerank_raw_delta_mean_abs": raw_delta.abs(),
                    "policy/top2_rerank_effective_delta_mean_abs": effective_delta.abs(),
                    "policy/top2_rerank_promote_second_mean": (effective_delta < 0).float(),
                }
            )
        if self.policy_logit_gain:
            top2 = torch.topk(logits, k=min(2, logits.size(-1)), dim=-1).values
            margin = top2[..., 0] - top2[..., 1] if top2.size(-1) > 1 else top2[..., 0]
            low_margin_gate = torch.sigmoid(
                (self.policy_logit_gain_threshold - margin) * self.policy_logit_gain_sharpness
            )
            learned_gate = torch.sigmoid(self.policy_gain_gate(core_output.pooled)).squeeze(-1)
            gate = low_margin_gate * learned_gate
            raw_gain = torch.tanh(self.policy_gain_head(core_output.pooled)) * self.policy_logit_gain_scale
            effective_gain = gate.unsqueeze(-1) * raw_gain
            logits = logits * (1.0 + effective_gain)
            metrics.update(
                {
                    "policy/logit_gain_gate_mean": gate,
                    "policy/logit_gain_gate_max": gate.max(),
                    "policy/logit_gain_low_margin_mean": low_margin_gate,
                    "policy/logit_gain_base_margin": margin,
                    "policy/logit_gain_raw_norm": raw_gain.norm(dim=-1),
                    "policy/logit_gain_effective_norm": effective_gain.norm(dim=-1),
                }
            )
        if self.policy_margin_residual:
            top2 = torch.topk(base_logits, k=min(2, base_logits.size(-1)), dim=-1).values
            margin = top2[..., 0] - top2[..., 1] if top2.size(-1) > 1 else top2[..., 0]
            low_margin_gate = torch.sigmoid((self.policy_margin_threshold - margin) * self.policy_margin_sharpness)
            learned_gate = torch.sigmoid(self.policy_residual_gate(core_output.pooled)).squeeze(-1)
            gate = low_margin_gate * learned_gate
            residual_logits = self.policy_residual_head(core_output.pooled) * self.policy_margin_residual_scale
            logits = base_logits + gate.unsqueeze(-1) * residual_logits
            metrics.update(
                {
                    "policy/margin_residual_gate_mean": gate,
                    "policy/margin_residual_gate_max": gate.max(),
                    "policy/margin_residual_low_margin_mean": low_margin_gate,
                    "policy/margin_residual_base_margin": margin,
                    "policy/margin_residual_logits_norm": residual_logits.norm(dim=-1),
                }
            )
        if self.policy_option_margin_adapter:
            option_margin_features = core_output.policy_features.get("option_margin_features")
            option_margin_stability = core_output.policy_features.get("option_margin_stability")
            if option_margin_features is not None and option_margin_stability is not None:
                top2 = torch.topk(logits, k=min(2, logits.size(-1)), dim=-1).values
                margin = top2[..., 0] - top2[..., 1] if top2.size(-1) > 1 else top2[..., 0]
                low_margin_gate = torch.sigmoid(
                    (self.policy_option_margin_threshold - margin) * self.policy_option_margin_sharpness
                )
                learned_gate = torch.sigmoid(self.policy_option_margin_gate(core_output.pooled)).squeeze(-1)
                gate = low_margin_gate * learned_gate * option_margin_stability.squeeze(-1)
                raw_option_delta = torch.tanh(self.policy_option_margin_adapter_head(option_margin_features))
                raw_option_delta = raw_option_delta * self.policy_option_margin_adapter_scale
                effective_option_delta = gate.unsqueeze(-1) * raw_option_delta
                logits = logits + effective_option_delta
                metrics.update(
                    {
                        "policy/option_margin_adapter_gate_mean": gate,
                        "policy/option_margin_adapter_gate_max": gate.max(),
                        "policy/option_margin_adapter_low_margin_mean": low_margin_gate,
                        "policy/option_margin_adapter_margin_before": margin,
                        "policy/option_margin_adapter_stability_mean": option_margin_stability.squeeze(-1),
                        "policy/option_margin_adapter_raw_norm": raw_option_delta.norm(dim=-1),
                        "policy/option_margin_adapter_effective_norm": effective_option_delta.norm(dim=-1),
                    }
                )
        if self.policy_option_top2_rerank and logits.size(-1) > 1:
            option_top2_features = core_output.policy_features.get("option_top2_features")
            option_top2_stability = core_output.policy_features.get("option_top2_stability")
            option_top2_duration_gate = core_output.policy_features.get("option_top2_duration_gate")
            if option_top2_features is not None and option_top2_stability is not None:
                top2 = torch.topk(logits, k=2, dim=-1)
                top2_values = top2.values
                top2_indices = top2.indices
                margin = top2_values[..., 0] - top2_values[..., 1]
                low_margin_gate = torch.sigmoid(
                    (self.policy_option_top2_rerank_threshold - margin) * self.policy_option_top2_rerank_sharpness
                )
                if self.policy_option_top2_use_duration_gate and option_top2_duration_gate is not None:
                    option_gate_signal = option_top2_duration_gate.squeeze(-1)
                else:
                    option_gate_signal = option_top2_stability.squeeze(-1)
                gate = low_margin_gate * option_gate_signal
                top1_embed = self.policy_option_top2_action_embed(top2_indices[..., 0])
                top2_embed = self.policy_option_top2_action_embed(top2_indices[..., 1])
                rerank_features = torch.cat(
                    [
                        option_top2_features,
                        top1_embed,
                        top2_embed,
                        top2_values,
                    ],
                    dim=-1,
                )
                raw_delta = torch.tanh(self.policy_option_top2_head(rerank_features)).squeeze(-1)
                raw_delta = raw_delta * self.policy_option_top2_rerank_scale
                effective_delta = gate * raw_delta
                effective_delta = effective_delta.to(logits.dtype)
                correction = torch.zeros_like(logits)
                correction.scatter_add_(1, top2_indices[..., 0].unsqueeze(-1), effective_delta.unsqueeze(-1))
                correction.scatter_add_(1, top2_indices[..., 1].unsqueeze(-1), (-effective_delta).unsqueeze(-1))
                logits = logits + correction
                top2_after = torch.topk(logits, k=2, dim=-1).values
                metrics.update(
                    {
                        "policy/option_top2_rerank_gate_mean": gate,
                        "policy/option_top2_rerank_gate_max": gate.max(),
                        "policy/option_top2_rerank_low_margin_mean": low_margin_gate,
                        "policy/option_top2_rerank_margin_before": margin,
                        "policy/option_top2_rerank_margin_after": top2_after[..., 0] - top2_after[..., 1],
                        "policy/option_top2_rerank_stability_mean": option_top2_stability.squeeze(-1),
                        "policy/option_top2_rerank_gate_signal_mean": option_gate_signal,
                        "policy/option_top2_rerank_raw_delta_mean_abs": raw_delta.abs(),
                        "policy/option_top2_rerank_effective_delta_mean_abs": effective_delta.abs(),
                        "policy/option_top2_rerank_promote_second_mean": (effective_delta < 0).float(),
                    }
                )
        value = self.value_head(core_output.pooled).squeeze(-1)
        return ModelOutput(
            logits=logits,
            value=value,
            metrics=metrics,
            next_state=core_output.next_state,
            aux_losses=core_output.aux_losses,
        )

    def get_dist(self, logits: torch.Tensor, temperature: float = 1.0) -> Categorical:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        return Categorical(logits=logits / temperature)

    def act(
        self,
        obs: TensorDict,
        state: TensorDict | None = None,
        done: torch.Tensor | None = None,
        greedy: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TensorDict, dict[str, torch.Tensor | float], dict[str, torch.Tensor]]:
        output = self.forward(obs, state=state, done=done)
        dist = self.get_dist(output.logits, temperature=temperature)
        if greedy:
            action = output.logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, output.value, output.next_state, output.metrics, output.aux_losses

    def evaluate_actions(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        state: TensorDict | None = None,
        done: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        output = self.forward(obs, state=state, done=done)
        dist = self.get_dist(output.logits)
        return {
            "log_prob": dist.log_prob(actions),
            "entropy": dist.entropy(),
            "value": output.value,
            "metrics": output.metrics,
            "next_state": output.next_state,
            "aux_losses": output.aux_losses,
        }
