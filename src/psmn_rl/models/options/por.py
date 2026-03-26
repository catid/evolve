from __future__ import annotations

import math

import torch
from torch import nn

from psmn_rl.models.common import CoreOutput, masked_mean
from psmn_rl.models.cores.dense import TransformerBlock
from psmn_rl.models.encoders import build_token_encoder


class PORCore(nn.Module):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        action_dim: int,
        option_count: int,
        termination_bias: float,
        option_action_adapter: bool,
        option_action_adapter_scale: float,
        option_action_adapter_min_duration: float,
        option_action_adapter_duration_sharpness: float,
        option_hidden_residual: bool,
        option_hidden_residual_scale: float,
        option_hidden_residual_min_duration: float,
        option_hidden_residual_duration_sharpness: float,
        option_action_experts: bool,
        option_action_experts_scale: float,
        option_action_experts_min_duration: float,
        option_action_experts_duration_sharpness: float,
        option_film: bool,
        option_film_scale: float,
        option_film_min_duration: float,
        option_film_duration_sharpness: float,
        option_context_film: bool,
        option_context_film_scale: float,
        option_context_film_min_duration: float,
        option_context_film_duration_sharpness: float,
        option_context_logits: bool,
        option_context_logits_scale: float,
        option_context_logits_min_duration: float,
        option_context_logits_duration_sharpness: float,
        option_context_gain: bool,
        option_context_gain_scale: float,
        option_context_gain_min_duration: float,
        option_context_gain_duration_sharpness: float,
    ) -> None:
        super().__init__()
        self.option_count = option_count
        self.termination_bias = termination_bias
        self.option_action_adapter = option_action_adapter
        self.option_action_adapter_scale = option_action_adapter_scale
        self.option_action_adapter_min_duration = option_action_adapter_min_duration
        self.option_action_adapter_duration_sharpness = option_action_adapter_duration_sharpness
        self.option_hidden_residual = option_hidden_residual
        self.option_hidden_residual_scale = option_hidden_residual_scale
        self.option_hidden_residual_min_duration = option_hidden_residual_min_duration
        self.option_hidden_residual_duration_sharpness = option_hidden_residual_duration_sharpness
        self.option_action_experts = option_action_experts
        self.option_action_experts_scale = option_action_experts_scale
        self.option_action_experts_min_duration = option_action_experts_min_duration
        self.option_action_experts_duration_sharpness = option_action_experts_duration_sharpness
        self.option_film = option_film
        self.option_film_scale = option_film_scale
        self.option_film_min_duration = option_film_min_duration
        self.option_film_duration_sharpness = option_film_duration_sharpness
        self.option_context_film = option_context_film
        self.option_context_film_scale = option_context_film_scale
        self.option_context_film_min_duration = option_context_film_min_duration
        self.option_context_film_duration_sharpness = option_context_film_duration_sharpness
        self.option_context_logits = option_context_logits
        self.option_context_logits_scale = option_context_logits_scale
        self.option_context_logits_min_duration = option_context_logits_min_duration
        self.option_context_logits_duration_sharpness = option_context_logits_duration_sharpness
        self.option_context_gain = option_context_gain
        self.option_context_gain_scale = option_context_gain_scale
        self.option_context_gain_min_duration = option_context_gain_min_duration
        self.option_context_gain_duration_sharpness = option_context_gain_duration_sharpness
        self.encoder = build_token_encoder(observation_space, token_dim, patch_size)
        self.input_proj = nn.Linear(token_dim, hidden_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.option_policy = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, option_count))
        self.option_embed = nn.Embedding(option_count, hidden_size)
        self.option_proj = nn.Linear(hidden_size, hidden_size)
        self.termination_head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        if self.option_action_adapter:
            self.option_action_head = nn.Sequential(
                nn.LayerNorm(option_count + 2),
                nn.Linear(option_count + 2, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, action_dim),
            )
        if self.option_hidden_residual:
            self.option_hidden_residual_head = nn.Sequential(
                nn.LayerNorm(hidden_size * 3 + 2),
                nn.Linear(hidden_size * 3 + 2, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
        if self.option_action_experts:
            self.option_action_expert_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Linear(hidden_size, hidden_size),
                        nn.GELU(),
                        nn.Linear(hidden_size, action_dim),
                    )
                    for _ in range(option_count)
                ]
            )
        if self.option_film:
            self.option_film_head = nn.Sequential(
                nn.LayerNorm(hidden_size * 3 + 2),
                nn.Linear(hidden_size * 3 + 2, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size * 2),
            )
        if self.option_context_film:
            self.option_context_film_head = nn.Sequential(
                nn.LayerNorm(hidden_size * 3 + 2),
                nn.Linear(hidden_size * 3 + 2, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size * 2),
            )
        if self.option_context_logits:
            self.option_context_logits_head = nn.Sequential(
                nn.LayerNorm(hidden_size * 2 + 2),
                nn.Linear(hidden_size * 2 + 2, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, action_dim),
            )
        self.norm = nn.LayerNorm(hidden_size)

    def initial_state(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        option_probs = torch.full((batch_size, self.option_count), 1.0 / self.option_count, device=device)
        duration = torch.zeros(batch_size, device=device)
        return {"option_probs": option_probs, "option_duration": duration}

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        for block in self.blocks:
            tokens = block(tokens)
        pooled = self.norm(masked_mean(tokens))

        prev_probs = state.get("option_probs")
        prev_duration = state.get("option_duration")
        if prev_probs is None or prev_duration is None:
            init_state = self.initial_state(pooled.size(0), pooled.device)
            prev_probs = init_state["option_probs"]
            prev_duration = init_state["option_duration"]

        if done is not None:
            done = done.float().unsqueeze(-1)
            uniform = torch.full_like(prev_probs, 1.0 / self.option_count)
            prev_probs = done * uniform + (1.0 - done) * prev_probs
            prev_duration = done.squeeze(-1) * 0.0 + (1.0 - done.squeeze(-1)) * prev_duration

        prev_context = prev_probs @ self.option_embed.weight
        proposal = torch.softmax(self.option_policy(pooled), dim=-1)
        terminate_prob = torch.sigmoid(self.termination_head(torch.cat([pooled, prev_context], dim=-1)) + self.termination_bias)
        next_probs = (1.0 - terminate_prob) * prev_probs + terminate_prob * proposal
        next_duration = (1.0 - terminate_prob.squeeze(-1)) * (prev_duration + 1.0) + terminate_prob.squeeze(-1)
        next_context = next_probs @ self.option_embed.weight

        prev_idx = prev_probs.argmax(dim=-1)
        next_idx = next_probs.argmax(dim=-1)
        switch_rate = (prev_idx != next_idx).float().mean()
        option_entropy = -(next_probs.clamp_min(1e-8) * next_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        logit_bias = None
        stability = None
        duration_gate = None
        entropy_norm = None
        if (
            self.option_action_adapter
            or self.option_hidden_residual
            or self.option_action_experts
            or self.option_film
            or self.option_context_film
            or self.option_context_logits
            or self.option_context_gain
        ):
            entropy = -(next_probs.clamp_min(1e-8) * next_probs.clamp_min(1e-8).log()).sum(dim=-1)
            entropy_norm = entropy / math.log(max(self.option_count, 2))
            if self.option_context_gain:
                gate_min_duration = self.option_context_gain_min_duration
                gate_sharpness = self.option_context_gain_duration_sharpness
            elif self.option_context_logits:
                gate_min_duration = self.option_context_logits_min_duration
                gate_sharpness = self.option_context_logits_duration_sharpness
            elif self.option_context_film:
                gate_min_duration = self.option_context_film_min_duration
                gate_sharpness = self.option_context_film_duration_sharpness
            elif self.option_film:
                gate_min_duration = self.option_film_min_duration
                gate_sharpness = self.option_film_duration_sharpness
            elif self.option_action_experts:
                gate_min_duration = self.option_action_experts_min_duration
                gate_sharpness = self.option_action_experts_duration_sharpness
            elif self.option_action_adapter:
                gate_min_duration = self.option_action_adapter_min_duration
                gate_sharpness = self.option_action_adapter_duration_sharpness
            else:
                gate_min_duration = self.option_hidden_residual_min_duration
                gate_sharpness = self.option_hidden_residual_duration_sharpness
            duration_gate = torch.sigmoid(
                (next_duration - gate_min_duration) * gate_sharpness
            )
            stability = duration_gate * (1.0 - entropy_norm.clamp(0.0, 1.0))
        context_delta = None
        if (
            self.option_hidden_residual
            or self.option_film
            or self.option_context_film
        ) and stability is not None and duration_gate is not None and entropy_norm is not None:
            context_delta = next_context - prev_context
        if self.option_context_film and context_delta is not None and stability is not None:
            context_film_features = torch.cat(
                [
                    pooled,
                    next_context,
                    context_delta,
                    duration_gate.unsqueeze(-1),
                    terminate_prob,
                ],
                dim=-1,
            )
            raw_context_film = self.option_context_film_head(context_film_features)
            raw_context_scale, raw_context_shift = raw_context_film.chunk(2, dim=-1)
            context_film_gate = self.option_context_film_scale * stability.unsqueeze(-1)
            context_film_scale = torch.tanh(raw_context_scale) * context_film_gate
            context_film_shift = raw_context_shift * context_film_gate
            next_context = next_context * (1.0 + context_film_scale) + context_film_shift
        option_context_projected = self.option_proj(next_context)
        option_context_gain = None
        if self.option_context_gain and stability is not None:
            option_context_gain = 1.0 + (self.option_context_gain_scale * stability.unsqueeze(-1))
            option_context_projected = option_context_projected * option_context_gain
        if self.option_context_logits and stability is not None and duration_gate is not None:
            context_logits_features = torch.cat(
                [
                    pooled,
                    option_context_projected,
                    duration_gate.unsqueeze(-1),
                    terminate_prob,
                ],
                dim=-1,
            )
            raw_context_logits = self.option_context_logits_head(context_logits_features)
            context_logits_bias = raw_context_logits * (self.option_context_logits_scale * stability.unsqueeze(-1))
            logit_bias = context_logits_bias if logit_bias is None else (logit_bias + context_logits_bias)
        conditioned = pooled + option_context_projected
        if self.option_film and context_delta is not None and stability is not None:
            film_features = torch.cat(
                [
                    pooled,
                    next_context,
                    context_delta,
                    duration_gate.unsqueeze(-1),
                    terminate_prob,
                ],
                dim=-1,
            )
            raw_film = self.option_film_head(film_features)
            raw_scale, raw_shift = raw_film.chunk(2, dim=-1)
            film_gate = self.option_film_scale * stability.unsqueeze(-1)
            film_scale = torch.tanh(raw_scale) * film_gate
            film_shift = raw_shift * film_gate
            conditioned = conditioned * (1.0 + film_scale) + film_shift
        if self.option_hidden_residual and stability is not None and duration_gate is not None and entropy_norm is not None:
            hidden_features = torch.cat(
                [
                    pooled,
                    next_context,
                    context_delta,
                    duration_gate.unsqueeze(-1),
                    terminate_prob,
                ],
                dim=-1,
            )
            raw_hidden_residual = self.option_hidden_residual_head(hidden_features)
            hidden_residual = raw_hidden_residual * (self.option_hidden_residual_scale * stability.unsqueeze(-1))
            conditioned = conditioned + hidden_residual
        if self.option_action_adapter:
            adapter_features = torch.cat(
                [
                    next_probs,
                    duration_gate.unsqueeze(-1),
                    terminate_prob,
                ],
                dim=-1,
            )
            raw_adapter = self.option_action_head(adapter_features)
            logit_bias = raw_adapter * (self.option_action_adapter_scale * stability.unsqueeze(-1))
        if self.option_action_experts and stability is not None:
            expert_logits = torch.stack([head(conditioned) for head in self.option_action_expert_heads], dim=1)
            mixed_expert_logits = (next_probs.unsqueeze(-1) * expert_logits).sum(dim=1)
            expert_bias = mixed_expert_logits * (self.option_action_experts_scale * stability.unsqueeze(-1))
            logit_bias = expert_bias if logit_bias is None else (logit_bias + expert_bias)

        metrics = {
            "route_entropy": float(option_entropy.item()),
            "option_duration": float(next_duration.mean().item()),
            "option_switch_rate": float(switch_rate.item()),
            "avg_halting_probability": float(terminate_prob.mean().item()),
            "termination_bias": float(self.termination_bias),
            "active_compute_proxy": 1.0,
        }
        if self.option_action_adapter and logit_bias is not None:
            metrics.update(
                {
                    "policy/option_action_adapter_stability": stability.mean(),
                    "policy/option_action_adapter_duration_gate": duration_gate.mean(),
                    "policy/option_action_adapter_entropy_norm": entropy_norm.mean(),
                    "policy/option_action_adapter_logits_norm": raw_adapter.norm(dim=-1).mean(),
                    "policy/option_action_adapter_bias_norm": logit_bias.norm(dim=-1).mean(),
                }
            )
        if self.option_hidden_residual and stability is not None and duration_gate is not None and entropy_norm is not None:
            metrics.update(
                {
                    "policy/option_hidden_residual_stability": stability.mean(),
                    "policy/option_hidden_residual_duration_gate": duration_gate.mean(),
                    "policy/option_hidden_residual_entropy_norm": entropy_norm.mean(),
                    "policy/option_hidden_residual_context_delta_norm": context_delta.norm(dim=-1).mean(),
                    "policy/option_hidden_residual_norm": hidden_residual.norm(dim=-1).mean(),
                }
            )
        if self.option_action_experts and stability is not None and duration_gate is not None and entropy_norm is not None:
            metrics.update(
                {
                    "policy/option_action_experts_stability": stability.mean(),
                    "policy/option_action_experts_duration_gate": duration_gate.mean(),
                    "policy/option_action_experts_entropy_norm": entropy_norm.mean(),
                    "policy/option_action_experts_logits_norm": mixed_expert_logits.norm(dim=-1).mean(),
                    "policy/option_action_experts_bias_norm": expert_bias.norm(dim=-1).mean(),
                }
            )
        if self.option_film and stability is not None and duration_gate is not None and entropy_norm is not None:
            metrics.update(
                {
                    "policy/option_film_stability": stability.mean(),
                    "policy/option_film_duration_gate": duration_gate.mean(),
                    "policy/option_film_entropy_norm": entropy_norm.mean(),
                    "policy/option_film_scale_norm": film_scale.norm(dim=-1).mean(),
                    "policy/option_film_shift_norm": film_shift.norm(dim=-1).mean(),
                }
            )
        if self.option_context_film and stability is not None and duration_gate is not None and entropy_norm is not None:
            metrics.update(
                {
                    "policy/option_context_film_stability": stability.mean(),
                    "policy/option_context_film_duration_gate": duration_gate.mean(),
                    "policy/option_context_film_entropy_norm": entropy_norm.mean(),
                    "policy/option_context_film_scale_norm": context_film_scale.norm(dim=-1).mean(),
                    "policy/option_context_film_shift_norm": context_film_shift.norm(dim=-1).mean(),
                }
            )
        if self.option_context_logits and stability is not None and duration_gate is not None and entropy_norm is not None:
            metrics.update(
                {
                    "policy/option_context_logits_stability": stability.mean(),
                    "policy/option_context_logits_duration_gate": duration_gate.mean(),
                    "policy/option_context_logits_entropy_norm": entropy_norm.mean(),
                    "policy/option_context_logits_norm": raw_context_logits.norm(dim=-1).mean(),
                    "policy/option_context_logits_bias_norm": context_logits_bias.norm(dim=-1).mean(),
                }
            )
        if self.option_context_gain and stability is not None and duration_gate is not None and entropy_norm is not None:
            metrics.update(
                {
                    "policy/option_context_gain_stability": stability.mean(),
                    "policy/option_context_gain_duration_gate": duration_gate.mean(),
                    "policy/option_context_gain_entropy_norm": entropy_norm.mean(),
                    "policy/option_context_gain_mean": option_context_gain.mean(),
                    "policy/option_context_gain_proj_norm": option_context_projected.norm(dim=-1).mean(),
                }
            )
        for option_index, value in enumerate(next_probs.mean(dim=0)):
            metrics[f"expert_load_{option_index}"] = float(value.item())

        next_state = {"option_probs": next_probs.detach(), "option_duration": next_duration.detach()}
        return CoreOutput(
            pooled=conditioned,
            tokens=tokens,
            metrics=metrics,
            next_state=next_state,
            logit_bias=logit_bias,
        )
