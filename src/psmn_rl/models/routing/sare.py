from __future__ import annotations

import math

import torch
from torch import nn

from psmn_rl.metrics import reduce_path_statistics
from psmn_rl.models.common import CoreOutput, entropy_from_probs, masked_mean, token_representation_metrics
from psmn_rl.models.cores.experts import ExpertBank
from psmn_rl.models.encoders import build_token_encoder


def _gather_expert_outputs(expert_outputs: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
    hidden_size = expert_outputs.size(-1)
    gather_index = topk_idx.unsqueeze(-1).expand(*topk_idx.shape, hidden_size)
    return torch.gather(expert_outputs, dim=2, index=gather_index)


def _route_metrics(route_probs: torch.Tensor, topk_idx: torch.Tensor, expert_count: int) -> dict[str, float]:
    flat_top1 = topk_idx[..., 0].reshape(-1)
    counts = torch.bincount(flat_top1, minlength=expert_count)
    stats = reduce_path_statistics(counts)
    metrics: dict[str, float] = {
        "route_entropy": float(entropy_from_probs(route_probs).mean().item()),
        "active_compute_proxy": float(topk_idx.shape[-1] / max(expert_count, 1)),
        **stats,
    }
    load = route_probs.mean(dim=(0, 1))
    for expert_index, value in enumerate(load):
        metrics[f"expert_load_{expert_index}"] = float(value.item())
    return metrics


class RoutedExpertCore(nn.Module):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
    ) -> None:
        super().__init__()
        self.expert_count = expert_count
        self.top_k = top_k
        self.temperature = temperature
        self.encoder = build_token_encoder(observation_space, token_dim, patch_size)
        self.input_proj = nn.Linear(token_dim, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.expert_keys = nn.Parameter(torch.randn(expert_count, hidden_size) / math.sqrt(hidden_size))
        self.bank = ExpertBank(hidden_size, expert_hidden_size, expert_count)
        self.output_norm = nn.LayerNorm(hidden_size)

    def route(
        self,
        tokens: torch.Tensor,
        route_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.query(tokens)
        scores = query @ self.expert_keys.t() / math.sqrt(query.size(-1))
        if route_bias is not None:
            scores = scores + route_bias.to(dtype=scores.dtype).unsqueeze(1)
        route_probs = torch.softmax(scores / self.temperature, dim=-1)
        topk_values, topk_idx = torch.topk(route_probs, k=min(self.top_k, self.expert_count), dim=-1)
        topk_values = topk_values / topk_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return route_probs, topk_values, topk_idx

    def apply_experts(self, tokens: torch.Tensor, topk_values: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        expert_outputs = self.bank.forward_all(tokens)
        gathered = _gather_expert_outputs(expert_outputs, topk_idx)
        mixed = (gathered * topk_values.unsqueeze(-1)).sum(dim=2)
        return mixed

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        route_probs, topk_values, topk_idx = self.route(tokens)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        metrics = {**_route_metrics(route_probs, topk_idx, self.expert_count), **token_representation_metrics(tokens, pooled)}
        return CoreOutput(pooled=pooled, tokens=tokens, metrics=metrics, next_state={})


class RoutedExpertPhaseMemoryCore(RoutedExpertCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
        )
        self.hidden_size = hidden_size
        self.memory_mix = float(memory_mix)
        self.memory_cell = nn.GRUCell(hidden_size, hidden_size)

    def initial_state(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        return {"hidden": torch.zeros(batch_size, self.hidden_size, device=device)}

    def _prepare_hidden(
        self,
        state: dict[str, torch.Tensor],
        pooled: torch.Tensor,
        done: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden = state.get("hidden")
        if hidden is None:
            hidden = torch.zeros(pooled.size(0), self.hidden_size, device=pooled.device, dtype=pooled.dtype)
        if done is not None:
            hidden = hidden * (~done).to(dtype=pooled.dtype).unsqueeze(-1)
        return hidden

    def _apply_memory(
        self,
        pooled: torch.Tensor,
        hidden: torch.Tensor,
        route_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        del route_probs
        next_hidden = self.memory_cell(pooled, hidden)
        next_hidden_for_mix = next_hidden.to(dtype=pooled.dtype)
        mixed_pooled = pooled + float(self.memory_mix) * (next_hidden_for_mix - pooled)
        metrics = {
            "memory/hidden_norm": float(next_hidden.norm(dim=-1).mean().item()),
            "memory/mix": self.memory_mix,
        }
        return mixed_pooled, next_hidden, metrics

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        route_probs, topk_values, topk_idx = self.route(tokens)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        hidden = self._prepare_hidden(state, pooled, done)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertGatedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        memory_gate_bias: float,
        memory_reset_bias: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_context = nn.Linear(expert_count, hidden_size, bias=False)
        self.memory_input = nn.Linear(hidden_size * 2, hidden_size)
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.reset_gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.constant_(self.update_gate[-1].bias, float(memory_gate_bias))
        nn.init.constant_(self.reset_gate[-1].bias, float(memory_reset_bias))

    def _apply_memory(
        self,
        pooled: torch.Tensor,
        hidden: torch.Tensor,
        route_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        route_summary = route_probs.mean(dim=1)
        route_context = torch.tanh(self.route_context(route_summary))

        gate_features = torch.cat([pooled, hidden, route_context], dim=-1)
        reset_gate = torch.sigmoid(self.reset_gate(gate_features))
        reset_hidden = hidden * (1.0 - reset_gate)

        memory_input = torch.cat([pooled, route_context], dim=-1)
        next_hidden = self.memory_cell(self.memory_input(memory_input), reset_hidden)

        update_features = torch.cat([pooled, next_hidden, route_context], dim=-1)
        update_gate = torch.sigmoid(self.update_gate(update_features))
        next_hidden_for_mix = next_hidden.to(dtype=pooled.dtype)
        effective_mix = (self.memory_mix * update_gate).to(dtype=pooled.dtype)
        mixed_pooled = pooled + effective_mix * (next_hidden_for_mix - pooled)
        metrics = {
            "memory/hidden_norm": float(next_hidden.norm(dim=-1).mean().item()),
            "memory/mix": float(effective_mix.mean().item()),
            "memory/update_gate_mean": float(update_gate.mean().item()),
            "memory/reset_gate_mean": float(reset_gate.mean().item()),
            "memory/route_context_norm": float(route_context.norm(dim=-1).mean().item()),
        }
        return mixed_pooled, next_hidden, metrics


class RoutedExpertRouteBiasedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        route_bias = torch.tanh(self.route_bias_proj(hidden)) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        route_memory_query = self.query(hidden)
        route_bias_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        route_bias_logits = base_route_logits + keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualPredictivePhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        token_summary = masked_mean(tokens)
        hidden = self._prepare_hidden(
            state,
            token_summary,
            done,
        )
        predictive_hidden = self.memory_cell(token_summary, hidden)
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(predictive_hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        route_bias_logits = base_route_logits + keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/predictive_hidden_norm": float(predictive_hidden.norm(dim=-1).mean().item()),
            "memory/keyed_predictive_delta_norm": float((predictive_hidden - hidden).norm(dim=-1).mean().item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualPredictiveBlendPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        predictive_blend: float = 0.5,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.predictive_blend = float(predictive_blend)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        token_summary = masked_mean(tokens)
        hidden = self._prepare_hidden(
            state,
            token_summary,
            done,
        )
        predictive_hidden = self.memory_cell(token_summary, hidden)
        blended_hidden = torch.lerp(hidden, predictive_hidden.to(dtype=hidden.dtype), self.predictive_blend)
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(blended_hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        route_bias_logits = base_route_logits + keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/predictive_hidden_norm": float(predictive_hidden.norm(dim=-1).mean().item()),
            "memory/predictive_blend_hidden_norm": float(blended_hidden.norm(dim=-1).mean().item()),
            "memory/keyed_predictive_delta_norm": float((predictive_hidden - hidden).norm(dim=-1).mean().item()),
            "memory/keyed_predictive_blend": self.predictive_blend,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualPredictiveCappedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        predictive_cap_ratio: float = 0.5,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.predictive_cap_ratio = float(predictive_cap_ratio)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        token_summary = masked_mean(tokens)
        hidden = self._prepare_hidden(
            state,
            token_summary,
            done,
        )
        predictive_hidden = self.memory_cell(token_summary, hidden)
        predictive_hidden_for_logits = predictive_hidden.to(dtype=hidden.dtype)

        base_route_logits = self.route_bias_proj(hidden)
        prior_route_memory_query = self.query(hidden)
        predictive_route_memory_query = self.query(predictive_hidden_for_logits)
        prior_keyed_route_logits = prior_route_memory_query @ self.expert_keys.t() / math.sqrt(prior_route_memory_query.size(-1))
        predictive_keyed_route_logits = predictive_route_memory_query @ self.expert_keys.t() / math.sqrt(predictive_route_memory_query.size(-1))
        predictive_delta_logits = predictive_keyed_route_logits - prior_keyed_route_logits
        prior_keyed_norm = prior_keyed_route_logits.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        predictive_delta_norm = predictive_delta_logits.norm(dim=-1, keepdim=True)
        max_delta_norm = prior_keyed_norm * self.predictive_cap_ratio
        predictive_delta_scale = torch.minimum(
            torch.ones_like(predictive_delta_norm),
            max_delta_norm / predictive_delta_norm.clamp_min(1e-6),
        )
        capped_predictive_delta_logits = predictive_delta_logits * predictive_delta_scale
        keyed_route_logits = prior_keyed_route_logits + capped_predictive_delta_logits
        route_bias_logits = base_route_logits + keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/prior_keyed_route_bias_logits_norm": float(prior_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/predictive_keyed_route_bias_logits_norm": float(predictive_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/capped_predictive_delta_logits_norm": float(capped_predictive_delta_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(predictive_route_memory_query.norm(dim=-1).mean().item()),
            "memory/predictive_hidden_norm": float(predictive_hidden_for_logits.norm(dim=-1).mean().item()),
            "memory/keyed_predictive_delta_norm": float(predictive_delta_logits.norm(dim=-1).mean().item()),
            "memory/predictive_delta_scale": float(predictive_delta_scale.mean().item()),
            "memory/keyed_predictive_cap_ratio": self.predictive_cap_ratio,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualPredictiveDeltaMixPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        predictive_delta_mix: float = 0.25,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.predictive_delta_mix = float(predictive_delta_mix)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        token_summary = masked_mean(tokens)
        hidden = self._prepare_hidden(
            state,
            token_summary,
            done,
        )
        predictive_hidden = self.memory_cell(token_summary, hidden)
        predictive_hidden_for_logits = predictive_hidden.to(dtype=hidden.dtype)

        base_route_logits = self.route_bias_proj(hidden)
        prior_route_memory_query = self.query(hidden)
        predictive_route_memory_query = self.query(predictive_hidden_for_logits)
        prior_keyed_route_logits = prior_route_memory_query @ self.expert_keys.t() / math.sqrt(prior_route_memory_query.size(-1))
        predictive_keyed_route_logits = predictive_route_memory_query @ self.expert_keys.t() / math.sqrt(predictive_route_memory_query.size(-1))
        predictive_delta_logits = predictive_keyed_route_logits - prior_keyed_route_logits
        mixed_predictive_delta_logits = predictive_delta_logits * self.predictive_delta_mix
        keyed_route_logits = prior_keyed_route_logits + mixed_predictive_delta_logits
        route_bias_logits = base_route_logits + keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/prior_keyed_route_bias_logits_norm": float(prior_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/predictive_keyed_route_bias_logits_norm": float(predictive_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/mixed_predictive_delta_logits_norm": float(mixed_predictive_delta_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(predictive_route_memory_query.norm(dim=-1).mean().item()),
            "memory/predictive_hidden_norm": float(predictive_hidden_for_logits.norm(dim=-1).mean().item()),
            "memory/keyed_predictive_delta_norm": float(predictive_delta_logits.norm(dim=-1).mean().item()),
            "memory/keyed_predictive_delta_mix": self.predictive_delta_mix,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualPredictivePositivePhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        token_summary = masked_mean(tokens)
        hidden = self._prepare_hidden(
            state,
            token_summary,
            done,
        )
        predictive_hidden = self.memory_cell(token_summary, hidden)
        predictive_hidden_for_logits = predictive_hidden.to(dtype=hidden.dtype)

        base_route_logits = self.route_bias_proj(hidden)
        prior_route_memory_query = self.query(hidden)
        predictive_route_memory_query = self.query(predictive_hidden_for_logits)
        prior_keyed_route_logits = prior_route_memory_query @ self.expert_keys.t() / math.sqrt(prior_route_memory_query.size(-1))
        predictive_keyed_route_logits = predictive_route_memory_query @ self.expert_keys.t() / math.sqrt(predictive_route_memory_query.size(-1))
        predictive_delta_logits = predictive_keyed_route_logits - prior_keyed_route_logits
        positive_predictive_delta_logits = torch.relu(predictive_delta_logits)
        keyed_route_logits = prior_keyed_route_logits + positive_predictive_delta_logits
        route_bias_logits = base_route_logits + keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/prior_keyed_route_bias_logits_norm": float(prior_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/predictive_keyed_route_bias_logits_norm": float(predictive_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/positive_predictive_delta_logits_norm": float(positive_predictive_delta_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(predictive_route_memory_query.norm(dim=-1).mean().item()),
            "memory/predictive_hidden_norm": float(predictive_hidden_for_logits.norm(dim=-1).mean().item()),
            "memory/keyed_predictive_delta_norm": float(predictive_delta_logits.norm(dim=-1).mean().item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualPredictiveAlignedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        token_summary = masked_mean(tokens)
        hidden = self._prepare_hidden(
            state,
            token_summary,
            done,
        )
        predictive_hidden = self.memory_cell(token_summary, hidden)
        predictive_hidden_for_logits = predictive_hidden.to(dtype=hidden.dtype)

        base_route_logits = self.route_bias_proj(hidden)
        prior_route_memory_query = self.query(hidden)
        predictive_route_memory_query = self.query(predictive_hidden_for_logits)
        prior_keyed_route_logits = prior_route_memory_query @ self.expert_keys.t() / math.sqrt(prior_route_memory_query.size(-1))
        predictive_keyed_route_logits = predictive_route_memory_query @ self.expert_keys.t() / math.sqrt(predictive_route_memory_query.size(-1))
        predictive_delta_logits = predictive_keyed_route_logits - prior_keyed_route_logits
        prior_sign = torch.sign(prior_keyed_route_logits)
        aligned_predictive_delta_logits = torch.relu(predictive_delta_logits * prior_sign) * prior_sign
        keyed_route_logits = prior_keyed_route_logits + aligned_predictive_delta_logits
        route_bias_logits = base_route_logits + keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/prior_keyed_route_bias_logits_norm": float(prior_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/predictive_keyed_route_bias_logits_norm": float(predictive_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/aligned_predictive_delta_logits_norm": float(aligned_predictive_delta_logits.norm(dim=-1).mean().item()),
            "memory/aligned_predictive_delta_fraction": float((aligned_predictive_delta_logits.abs() > 0).float().mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(predictive_route_memory_query.norm(dim=-1).mean().item()),
            "memory/predictive_hidden_norm": float(predictive_hidden_for_logits.norm(dim=-1).mean().item()),
            "memory/keyed_predictive_delta_norm": float(predictive_delta_logits.norm(dim=-1).mean().item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualPredictiveWeakPriorReplacePhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        token_summary = masked_mean(tokens)
        hidden = self._prepare_hidden(
            state,
            token_summary,
            done,
        )
        predictive_hidden = self.memory_cell(token_summary, hidden)
        predictive_hidden_for_logits = predictive_hidden.to(dtype=hidden.dtype)

        base_route_logits = self.route_bias_proj(hidden)
        prior_route_memory_query = self.query(hidden)
        predictive_route_memory_query = self.query(predictive_hidden_for_logits)
        prior_keyed_route_logits = prior_route_memory_query @ self.expert_keys.t() / math.sqrt(prior_route_memory_query.size(-1))
        predictive_keyed_route_logits = predictive_route_memory_query @ self.expert_keys.t() / math.sqrt(predictive_route_memory_query.size(-1))
        prior_abs = prior_keyed_route_logits.abs()
        prior_scale = prior_abs.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        weak_prior_replace_gate = torch.clamp(1.0 - (prior_abs / (prior_scale * 1.5)), min=0.0, max=1.0)
        replaced_keyed_route_logits = predictive_keyed_route_logits * weak_prior_replace_gate
        kept_prior_keyed_route_logits = prior_keyed_route_logits * (1.0 - weak_prior_replace_gate)
        keyed_route_logits = kept_prior_keyed_route_logits + replaced_keyed_route_logits
        route_bias_logits = base_route_logits + keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/prior_keyed_route_bias_logits_norm": float(prior_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/predictive_keyed_route_bias_logits_norm": float(predictive_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/replaced_keyed_route_bias_logits_norm": float(replaced_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/kept_prior_keyed_route_bias_logits_norm": float(kept_prior_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/weak_prior_replace_gate": float(weak_prior_replace_gate.mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(predictive_route_memory_query.norm(dim=-1).mean().item()),
            "memory/predictive_hidden_norm": float(predictive_hidden_for_logits.norm(dim=-1).mean().item()),
            "memory/keyed_predictive_delta_norm": float((predictive_keyed_route_logits - prior_keyed_route_logits).norm(dim=-1).mean().item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualPredictiveWeakPriorAssistPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        token_summary = masked_mean(tokens)
        hidden = self._prepare_hidden(
            state,
            token_summary,
            done,
        )
        predictive_hidden = self.memory_cell(token_summary, hidden)
        predictive_hidden_for_logits = predictive_hidden.to(dtype=hidden.dtype)

        base_route_logits = self.route_bias_proj(hidden)
        prior_route_memory_query = self.query(hidden)
        predictive_route_memory_query = self.query(predictive_hidden_for_logits)
        prior_keyed_route_logits = prior_route_memory_query @ self.expert_keys.t() / math.sqrt(prior_route_memory_query.size(-1))
        predictive_keyed_route_logits = predictive_route_memory_query @ self.expert_keys.t() / math.sqrt(predictive_route_memory_query.size(-1))
        prior_abs = prior_keyed_route_logits.abs()
        prior_scale = prior_abs.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        weak_prior_assist_gate = torch.clamp(1.0 - (prior_abs / (prior_scale * 1.5)), min=0.0, max=1.0)
        assisted_predictive_keyed_route_logits = predictive_keyed_route_logits * weak_prior_assist_gate
        keyed_route_logits = prior_keyed_route_logits + assisted_predictive_keyed_route_logits
        route_bias_logits = base_route_logits + keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/prior_keyed_route_bias_logits_norm": float(prior_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/predictive_keyed_route_bias_logits_norm": float(predictive_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/assisted_predictive_keyed_route_bias_logits_norm": float(assisted_predictive_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/weak_prior_assist_gate": float(weak_prior_assist_gate.mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(predictive_route_memory_query.norm(dim=-1).mean().item()),
            "memory/predictive_hidden_norm": float(predictive_hidden_for_logits.norm(dim=-1).mean().item()),
            "memory/keyed_predictive_delta_norm": float((predictive_keyed_route_logits - prior_keyed_route_logits).norm(dim=-1).mean().item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualPredictiveWeakPriorTop1AssistPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        token_summary = masked_mean(tokens)
        hidden = self._prepare_hidden(
            state,
            token_summary,
            done,
        )
        predictive_hidden = self.memory_cell(token_summary, hidden)
        predictive_hidden_for_logits = predictive_hidden.to(dtype=hidden.dtype)

        base_route_logits = self.route_bias_proj(hidden)
        prior_route_memory_query = self.query(hidden)
        predictive_route_memory_query = self.query(predictive_hidden_for_logits)
        prior_keyed_route_logits = prior_route_memory_query @ self.expert_keys.t() / math.sqrt(prior_route_memory_query.size(-1))
        predictive_keyed_route_logits = predictive_route_memory_query @ self.expert_keys.t() / math.sqrt(predictive_route_memory_query.size(-1))
        prior_abs = prior_keyed_route_logits.abs()
        prior_scale = prior_abs.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        weak_prior_assist_gate = torch.clamp(1.0 - (prior_abs / (prior_scale * 1.5)), min=0.0, max=1.0)
        weakest_idx = prior_abs.argmin(dim=-1, keepdim=True)
        weakest_mask = torch.zeros_like(prior_keyed_route_logits).scatter_(-1, weakest_idx, 1.0)
        top1_weak_prior_assist_gate = weak_prior_assist_gate * weakest_mask
        assisted_predictive_keyed_route_logits = predictive_keyed_route_logits * top1_weak_prior_assist_gate
        keyed_route_logits = prior_keyed_route_logits + assisted_predictive_keyed_route_logits
        route_bias_logits = base_route_logits + keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/prior_keyed_route_bias_logits_norm": float(prior_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/predictive_keyed_route_bias_logits_norm": float(predictive_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/assisted_predictive_keyed_route_bias_logits_norm": float(assisted_predictive_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/weak_prior_assist_gate": float(weak_prior_assist_gate.mean().item()),
            "memory/top1_weak_prior_assist_gate": float(top1_weak_prior_assist_gate.mean().item()),
            "memory/top1_weak_prior_density": float(weakest_mask.mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(predictive_route_memory_query.norm(dim=-1).mean().item()),
            "memory/predictive_hidden_norm": float(predictive_hidden_for_logits.norm(dim=-1).mean().item()),
            "memory/keyed_predictive_delta_norm": float((predictive_keyed_route_logits - prior_keyed_route_logits).norm(dim=-1).mean().item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualPredictiveWeakPriorTop1MaxMergePhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        token_summary = masked_mean(tokens)
        hidden = self._prepare_hidden(
            state,
            token_summary,
            done,
        )
        predictive_hidden = self.memory_cell(token_summary, hidden)
        predictive_hidden_for_logits = predictive_hidden.to(dtype=hidden.dtype)

        base_route_logits = self.route_bias_proj(hidden)
        prior_route_memory_query = self.query(hidden)
        predictive_route_memory_query = self.query(predictive_hidden_for_logits)
        prior_keyed_route_logits = prior_route_memory_query @ self.expert_keys.t() / math.sqrt(prior_route_memory_query.size(-1))
        predictive_keyed_route_logits = predictive_route_memory_query @ self.expert_keys.t() / math.sqrt(predictive_route_memory_query.size(-1))
        prior_abs = prior_keyed_route_logits.abs()
        predictive_abs = predictive_keyed_route_logits.abs()
        prior_scale = prior_abs.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        weak_prior_merge_gate = torch.clamp(1.0 - (prior_abs / (prior_scale * 1.5)), min=0.0, max=1.0)
        weakest_idx = prior_abs.argmin(dim=-1, keepdim=True)
        weakest_mask = torch.zeros_like(prior_keyed_route_logits).scatter_(-1, weakest_idx, 1.0)
        predictive_wins = (predictive_abs > prior_abs).to(dtype=prior_keyed_route_logits.dtype)
        eligible_weakest_mask = weakest_mask * (weak_prior_merge_gate > 0.0).to(dtype=prior_keyed_route_logits.dtype)
        top1_weak_prior_max_merge_gate = eligible_weakest_mask * predictive_wins
        maxmerged_predictive_keyed_route_logits = predictive_keyed_route_logits * top1_weak_prior_max_merge_gate
        kept_prior_keyed_route_logits = prior_keyed_route_logits * (1.0 - top1_weak_prior_max_merge_gate)
        keyed_route_logits = kept_prior_keyed_route_logits + maxmerged_predictive_keyed_route_logits
        route_bias_logits = base_route_logits + keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/prior_keyed_route_bias_logits_norm": float(prior_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/predictive_keyed_route_bias_logits_norm": float(predictive_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/maxmerged_predictive_keyed_route_bias_logits_norm": float(maxmerged_predictive_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/weak_prior_merge_gate": float(weak_prior_merge_gate.mean().item()),
            "memory/top1_weak_prior_max_merge_gate": float(top1_weak_prior_max_merge_gate.mean().item()),
            "memory/top1_weak_prior_density": float(weakest_mask.mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(predictive_route_memory_query.norm(dim=-1).mean().item()),
            "memory/predictive_hidden_norm": float(predictive_hidden_for_logits.norm(dim=-1).mean().item()),
            "memory/keyed_predictive_delta_norm": float((predictive_keyed_route_logits - prior_keyed_route_logits).norm(dim=-1).mean().item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualPredictiveTop1ConsensusBonusPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        consensus_bonus_scale: float = 0.5,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.consensus_bonus_scale = float(consensus_bonus_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        token_summary = masked_mean(tokens)
        hidden = self._prepare_hidden(
            state,
            token_summary,
            done,
        )
        predictive_hidden = self.memory_cell(token_summary, hidden)
        predictive_hidden_for_logits = predictive_hidden.to(dtype=hidden.dtype)

        base_route_logits = self.route_bias_proj(hidden)
        prior_route_memory_query = self.query(hidden)
        predictive_route_memory_query = self.query(predictive_hidden_for_logits)
        prior_keyed_route_logits = prior_route_memory_query @ self.expert_keys.t() / math.sqrt(prior_route_memory_query.size(-1))
        predictive_keyed_route_logits = predictive_route_memory_query @ self.expert_keys.t() / math.sqrt(predictive_route_memory_query.size(-1))
        prior_top1_idx = prior_keyed_route_logits.argmax(dim=-1, keepdim=True)
        predictive_top1_idx = predictive_keyed_route_logits.argmax(dim=-1, keepdim=True)
        top1_consensus = (prior_top1_idx == predictive_top1_idx).to(dtype=prior_keyed_route_logits.dtype)
        prior_top1_mask = torch.zeros_like(prior_keyed_route_logits).scatter_(-1, prior_top1_idx, 1.0)
        prior_top1_bonus = prior_keyed_route_logits * prior_top1_mask
        predictive_top1_bonus = predictive_keyed_route_logits * prior_top1_mask
        shared_positive_bonus = torch.minimum(
            torch.relu(prior_top1_bonus),
            torch.relu(predictive_top1_bonus),
        )
        consensus_bonus_logits = shared_positive_bonus * top1_consensus * self.consensus_bonus_scale
        keyed_route_logits = prior_keyed_route_logits + consensus_bonus_logits
        route_bias_logits = base_route_logits + keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/prior_keyed_route_bias_logits_norm": float(prior_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/predictive_keyed_route_bias_logits_norm": float(predictive_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/consensus_bonus_logits_norm": float(consensus_bonus_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/top1_consensus_rate": float(top1_consensus.mean().item()),
            "memory/top1_consensus_density": float((prior_top1_mask * top1_consensus).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(predictive_route_memory_query.norm(dim=-1).mean().item()),
            "memory/predictive_hidden_norm": float(predictive_hidden_for_logits.norm(dim=-1).mean().item()),
            "memory/keyed_predictive_delta_norm": float((predictive_keyed_route_logits - prior_keyed_route_logits).norm(dim=-1).mean().item()),
            "memory/consensus_bonus_scale": self.consensus_bonus_scale,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualPredictiveTop1DisagreementBonusPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        disagreement_bonus_scale: float = 0.5,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.disagreement_bonus_scale = float(disagreement_bonus_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        token_summary = masked_mean(tokens)
        hidden = self._prepare_hidden(
            state,
            token_summary,
            done,
        )
        predictive_hidden = self.memory_cell(token_summary, hidden)
        predictive_hidden_for_logits = predictive_hidden.to(dtype=hidden.dtype)

        base_route_logits = self.route_bias_proj(hidden)
        prior_route_memory_query = self.query(hidden)
        predictive_route_memory_query = self.query(predictive_hidden_for_logits)
        prior_keyed_route_logits = prior_route_memory_query @ self.expert_keys.t() / math.sqrt(prior_route_memory_query.size(-1))
        predictive_keyed_route_logits = predictive_route_memory_query @ self.expert_keys.t() / math.sqrt(predictive_route_memory_query.size(-1))
        prior_top1_idx = prior_keyed_route_logits.argmax(dim=-1, keepdim=True)
        predictive_top1_idx = predictive_keyed_route_logits.argmax(dim=-1, keepdim=True)
        top1_disagreement = (prior_top1_idx != predictive_top1_idx).to(dtype=prior_keyed_route_logits.dtype)
        predictive_top1_mask = torch.zeros_like(prior_keyed_route_logits).scatter_(-1, predictive_top1_idx, 1.0)
        predictive_top1_logits = predictive_keyed_route_logits * predictive_top1_mask
        prior_at_predictive_top1_logits = prior_keyed_route_logits * predictive_top1_mask
        challenger_margin_logits = torch.relu(predictive_top1_logits - prior_at_predictive_top1_logits)
        disagreement_bonus_logits = challenger_margin_logits * top1_disagreement * self.disagreement_bonus_scale
        keyed_route_logits = prior_keyed_route_logits + disagreement_bonus_logits
        route_bias_logits = base_route_logits + keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/prior_keyed_route_bias_logits_norm": float(prior_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/predictive_keyed_route_bias_logits_norm": float(predictive_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/disagreement_bonus_logits_norm": float(disagreement_bonus_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/top1_disagreement_rate": float(top1_disagreement.mean().item()),
            "memory/top1_disagreement_density": float((predictive_top1_mask * top1_disagreement).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(predictive_route_memory_query.norm(dim=-1).mean().item()),
            "memory/predictive_hidden_norm": float(predictive_hidden_for_logits.norm(dim=-1).mean().item()),
            "memory/keyed_predictive_delta_norm": float((predictive_keyed_route_logits - prior_keyed_route_logits).norm(dim=-1).mean().item()),
            "memory/disagreement_bonus_scale": self.disagreement_bonus_scale,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualPredictiveBaseTop1BonusPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        predictive_base_bonus_scale: float = 0.5,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.predictive_base_bonus_scale = float(predictive_base_bonus_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        token_summary = masked_mean(tokens)
        hidden = self._prepare_hidden(
            state,
            token_summary,
            done,
        )
        predictive_hidden = self.memory_cell(token_summary, hidden)
        predictive_hidden_for_logits = predictive_hidden.to(dtype=hidden.dtype)

        base_route_logits = self.route_bias_proj(hidden)
        predictive_base_route_logits = self.route_bias_proj(predictive_hidden_for_logits)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        base_top1_idx = base_route_logits.argmax(dim=-1, keepdim=True)
        predictive_base_top1_idx = predictive_base_route_logits.argmax(dim=-1, keepdim=True)
        predictive_base_top1_mask = torch.zeros_like(base_route_logits).scatter_(-1, predictive_base_top1_idx, 1.0)
        predictive_base_top1_logits = predictive_base_route_logits * predictive_base_top1_mask
        base_at_predictive_top1_logits = base_route_logits * predictive_base_top1_mask
        predictive_base_margin_logits = torch.relu(predictive_base_top1_logits - base_at_predictive_top1_logits)
        predictive_base_bonus_logits = predictive_base_margin_logits * self.predictive_base_bonus_scale
        keyed_route_logits = keyed_route_logits
        route_bias_logits = base_route_logits + keyed_route_logits + predictive_base_bonus_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        predictive_base_top1_disagreement = (base_top1_idx != predictive_base_top1_idx).to(dtype=base_route_logits.dtype)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/predictive_base_route_bias_logits_norm": float(predictive_base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/predictive_base_bonus_logits_norm": float(predictive_base_bonus_logits.norm(dim=-1).mean().item()),
            "memory/predictive_base_top1_disagreement_rate": float(predictive_base_top1_disagreement.mean().item()),
            "memory/predictive_base_top1_density": float(predictive_base_top1_mask.mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/predictive_hidden_norm": float(predictive_hidden_for_logits.norm(dim=-1).mean().item()),
            "memory/predictive_base_bonus_scale": self.predictive_base_bonus_scale,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualPredictiveBaseDeltaMixPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        predictive_base_delta_mix: float = 0.25,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.predictive_base_delta_mix = float(predictive_base_delta_mix)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        token_summary = masked_mean(tokens)
        hidden = self._prepare_hidden(
            state,
            token_summary,
            done,
        )
        predictive_hidden = self.memory_cell(token_summary, hidden)
        predictive_hidden_for_logits = predictive_hidden.to(dtype=hidden.dtype)

        base_route_logits = self.route_bias_proj(hidden)
        predictive_base_route_logits = self.route_bias_proj(predictive_hidden_for_logits)
        predictive_base_delta_logits = predictive_base_route_logits - base_route_logits
        mixed_predictive_base_delta_logits = predictive_base_delta_logits * self.predictive_base_delta_mix
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        route_bias_logits = base_route_logits + mixed_predictive_base_delta_logits + keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/predictive_base_route_bias_logits_norm": float(predictive_base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/predictive_base_delta_logits_norm": float(predictive_base_delta_logits.norm(dim=-1).mean().item()),
            "memory/mixed_predictive_base_delta_logits_norm": float(mixed_predictive_base_delta_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/predictive_hidden_norm": float(predictive_hidden_for_logits.norm(dim=-1).mean().item()),
            "memory/predictive_base_delta_mix": self.predictive_base_delta_mix,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualPredictiveBaseDeltaEntropyGatedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        predictive_base_delta_mix: float = 0.25,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.predictive_base_delta_mix = float(predictive_base_delta_mix)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        token_summary = masked_mean(tokens)
        hidden = self._prepare_hidden(
            state,
            token_summary,
            done,
        )
        predictive_hidden = self.memory_cell(token_summary, hidden)
        predictive_hidden_for_logits = predictive_hidden.to(dtype=hidden.dtype)

        base_route_logits = self.route_bias_proj(hidden)
        predictive_base_route_logits = self.route_bias_proj(predictive_hidden_for_logits)
        predictive_base_delta_logits = predictive_base_route_logits - base_route_logits
        base_route_probs = torch.softmax(base_route_logits / max(self.temperature, 1e-6), dim=-1)
        base_route_entropy = -(base_route_probs * torch.log(base_route_probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
        normalized_uncertainty = base_route_entropy / math.log(self.expert_count)
        mixed_predictive_base_delta_logits = predictive_base_delta_logits * normalized_uncertainty * self.predictive_base_delta_mix
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        route_bias_logits = base_route_logits + mixed_predictive_base_delta_logits + keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/predictive_base_route_bias_logits_norm": float(predictive_base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/predictive_base_delta_logits_norm": float(predictive_base_delta_logits.norm(dim=-1).mean().item()),
            "memory/mixed_predictive_base_delta_logits_norm": float(mixed_predictive_base_delta_logits.norm(dim=-1).mean().item()),
            "memory/predictive_base_uncertainty_gate": float(normalized_uncertainty.mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/predictive_hidden_norm": float(predictive_hidden_for_logits.norm(dim=-1).mean().item()),
            "memory/predictive_base_delta_mix": self.predictive_base_delta_mix,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualPredictiveBaseDeltaWeakBaseGatedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        predictive_base_delta_mix: float = 0.25,
        predictive_base_target_base_norm: float = 1.0,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.predictive_base_delta_mix = float(predictive_base_delta_mix)
        self.predictive_base_target_base_norm = float(predictive_base_target_base_norm)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        token_summary = masked_mean(tokens)
        hidden = self._prepare_hidden(
            state,
            token_summary,
            done,
        )
        predictive_hidden = self.memory_cell(token_summary, hidden)
        predictive_hidden_for_logits = predictive_hidden.to(dtype=hidden.dtype)

        base_route_logits = self.route_bias_proj(hidden)
        predictive_base_route_logits = self.route_bias_proj(predictive_hidden_for_logits)
        predictive_base_delta_logits = predictive_base_route_logits - base_route_logits
        base_route_logits_norm = base_route_logits.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        weak_base_gate = torch.clamp(
            (self.predictive_base_target_base_norm / base_route_logits_norm) - 1.0,
            min=0.0,
            max=1.0,
        )
        mixed_predictive_base_delta_logits = predictive_base_delta_logits * weak_base_gate * self.predictive_base_delta_mix
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        route_bias_logits = base_route_logits + mixed_predictive_base_delta_logits + keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/predictive_base_route_bias_logits_norm": float(predictive_base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/predictive_base_delta_logits_norm": float(predictive_base_delta_logits.norm(dim=-1).mean().item()),
            "memory/mixed_predictive_base_delta_logits_norm": float(mixed_predictive_base_delta_logits.norm(dim=-1).mean().item()),
            "memory/predictive_base_weak_gate": float(weak_base_gate.mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/predictive_hidden_norm": float(predictive_hidden_for_logits.norm(dim=-1).mean().item()),
            "memory/predictive_base_delta_mix": self.predictive_base_delta_mix,
            "memory/predictive_base_target_base_norm": self.predictive_base_target_base_norm,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualPredictiveDeltaGatedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        token_summary = masked_mean(tokens)
        hidden = self._prepare_hidden(
            state,
            token_summary,
            done,
        )
        predictive_hidden = self.memory_cell(token_summary, hidden)
        predictive_hidden_for_logits = predictive_hidden.to(dtype=hidden.dtype)

        base_route_logits = self.route_bias_proj(hidden)
        prior_route_memory_query = self.query(hidden)
        predictive_route_memory_query = self.query(predictive_hidden_for_logits)
        prior_keyed_route_logits = prior_route_memory_query @ self.expert_keys.t() / math.sqrt(prior_route_memory_query.size(-1))
        predictive_keyed_route_logits = predictive_route_memory_query @ self.expert_keys.t() / math.sqrt(predictive_route_memory_query.size(-1))
        predictive_delta_logits = predictive_keyed_route_logits - prior_keyed_route_logits
        positive_predictive_delta_logits = torch.relu(predictive_delta_logits)
        prior_keyed_scale = prior_keyed_route_logits.abs().mean(dim=-1, keepdim=True)
        positive_delta_scale = positive_predictive_delta_logits.abs().mean(dim=-1, keepdim=True)
        predictive_delta_gate = positive_delta_scale / (positive_delta_scale + prior_keyed_scale + 1e-6)
        gated_positive_predictive_delta_logits = positive_predictive_delta_logits * predictive_delta_gate
        keyed_route_logits = prior_keyed_route_logits + gated_positive_predictive_delta_logits
        route_bias_logits = base_route_logits + keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/prior_keyed_route_bias_logits_norm": float(prior_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/predictive_keyed_route_bias_logits_norm": float(predictive_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/positive_predictive_delta_logits_norm": float(positive_predictive_delta_logits.norm(dim=-1).mean().item()),
            "memory/gated_positive_predictive_delta_logits_norm": float(gated_positive_predictive_delta_logits.norm(dim=-1).mean().item()),
            "memory/predictive_delta_gate": float(predictive_delta_gate.mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(predictive_route_memory_query.norm(dim=-1).mean().item()),
            "memory/predictive_hidden_norm": float(predictive_hidden_for_logits.norm(dim=-1).mean().item()),
            "memory/keyed_predictive_delta_norm": float(predictive_delta_logits.norm(dim=-1).mean().item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualEmaPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        keyed_ema_decay: float = 0.75,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.keyed_ema_decay = float(keyed_ema_decay)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        ema_hidden = state.get("ema_hidden")
        if ema_hidden is None:
            ema_hidden = hidden
        else:
            ema_hidden = ema_hidden.to(device=hidden.device, dtype=hidden.dtype)
            if done is not None:
                ema_hidden = ema_hidden * (~done).to(dtype=hidden.dtype).unsqueeze(-1)

        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(ema_hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        route_bias_logits = base_route_logits + keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        next_hidden_for_ema = next_hidden.to(dtype=ema_hidden.dtype)
        next_ema_hidden = self.keyed_ema_decay * ema_hidden + (1.0 - self.keyed_ema_decay) * next_hidden_for_ema
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/ema_hidden_norm": float(next_ema_hidden.norm(dim=-1).mean().item()),
            "memory/keyed_ema_delta_norm": float((next_ema_hidden - hidden).norm(dim=-1).mean().item()),
            "memory/keyed_ema_decay": self.keyed_ema_decay,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach(), "ema_hidden": next_ema_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualOrthogonalPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        base_norm_sq = base_route_logits.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        alignment_weight = (keyed_route_logits * base_route_logits).sum(dim=-1, keepdim=True) / base_norm_sq
        aligned_keyed_route_logits = alignment_weight * base_route_logits
        orthogonal_keyed_route_logits = keyed_route_logits - aligned_keyed_route_logits
        route_bias_logits = base_route_logits + orthogonal_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        keyed_norm = keyed_route_logits.norm(dim=-1)
        base_norm = base_route_logits.norm(dim=-1)
        alignment_cosine = (keyed_route_logits * base_route_logits).sum(dim=-1) / (keyed_norm * base_norm).clamp_min(1e-6)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_norm.mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_norm.mean().item()),
            "memory/aligned_keyed_route_bias_logits_norm": float(aligned_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/orthogonal_keyed_route_bias_logits_norm": float(orthogonal_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/keyed_route_alignment_cosine": float(alignment_cosine.mean().item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualBaseClippedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        target_base_norm: float = 0.8,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.target_base_norm = float(target_base_norm)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        base_norm = base_route_logits.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        base_clip_scale = (self.target_base_norm / base_norm).clamp(max=1.0)
        clipped_base_route_logits = base_route_logits * base_clip_scale
        route_bias_logits = clipped_base_route_logits + keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_norm.mean().item()),
            "memory/clipped_base_route_bias_logits_norm": float(clipped_base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/base_clip_scale_mean": float(base_clip_scale.mean().item()),
            "memory/base_clip_scale_min": float(base_clip_scale.min().item()),
            "memory/base_clip_scale_max": float(base_clip_scale.max().item()),
            "memory/base_clip_target_norm": self.target_base_norm,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualChallengerPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        challenger_scale: float = 0.5,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.challenger_scale = float(challenger_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))

        base_topk_idx = base_route_logits.topk(self.top_k, dim=-1).indices
        keyed_topk_idx = keyed_route_logits.topk(self.top_k, dim=-1).indices
        base_topk_mask = torch.zeros_like(base_route_logits, dtype=torch.bool).scatter(1, base_topk_idx, True)
        keyed_topk_mask = torch.zeros_like(keyed_route_logits, dtype=torch.bool).scatter(1, keyed_topk_idx, True)
        challenger_mask = keyed_topk_mask & ~base_topk_mask

        challenger_bonus = keyed_route_logits * challenger_mask.to(keyed_route_logits.dtype) * self.challenger_scale
        route_bias_logits = base_route_logits + keyed_route_logits + challenger_bonus
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/challenger_route_bias_logits_norm": float(challenger_bonus.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/base_topk_density": float(base_topk_mask.float().mean().item()),
            "memory/keyed_topk_density": float(keyed_topk_mask.float().mean().item()),
            "memory/challenger_mask_density": float(challenger_mask.float().mean().item()),
            "memory/challenger_scale": self.challenger_scale,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualRatioGatedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        ratio_threshold: float = 20.0,
        max_boost: float = 1.5,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.ratio_threshold = float(ratio_threshold)
        self.max_boost = float(max_boost)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        base_norm = base_route_logits.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        keyed_norm = keyed_route_logits.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        base_keyed_ratio = base_norm / keyed_norm
        ratio_gate = ((base_keyed_ratio - self.ratio_threshold) / self.ratio_threshold).clamp(0.0, 1.0)
        keyed_residual_scale = 1.0 + ratio_gate * (self.max_boost - 1.0)
        gated_keyed_route_logits = keyed_route_logits * keyed_residual_scale
        route_bias_logits = base_route_logits + gated_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_norm.mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_norm.mean().item()),
            "memory/gated_keyed_route_bias_logits_norm": float(gated_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/base_keyed_ratio_mean": float(base_keyed_ratio.mean().item()),
            "memory/ratio_gate_mean": float(ratio_gate.mean().item()),
            "memory/ratio_gate_min": float(ratio_gate.min().item()),
            "memory/ratio_gate_max": float(ratio_gate.max().item()),
            "memory/keyed_residual_scale_mean": float(keyed_residual_scale.mean().item()),
            "memory/keyed_residual_scale_min": float(keyed_residual_scale.min().item()),
            "memory/keyed_residual_scale_max": float(keyed_residual_scale.max().item()),
            "memory/keyed_residual_ratio_threshold": self.ratio_threshold,
            "memory/keyed_residual_max_boost": self.max_boost,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualTargetNormPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        target_keyed_norm: float = 0.05,
        min_scale: float = 0.75,
        max_scale: float = 1.75,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.target_keyed_norm = float(target_keyed_norm)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        keyed_norm = keyed_route_logits.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        keyed_residual_scale = (self.target_keyed_norm / keyed_norm).clamp(self.min_scale, self.max_scale)
        scaled_keyed_route_logits = keyed_route_logits * keyed_residual_scale
        route_bias_logits = base_route_logits + scaled_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_norm.mean().item()),
            "memory/scaled_keyed_route_bias_logits_norm": float(scaled_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/keyed_residual_scale_mean": float(keyed_residual_scale.mean().item()),
            "memory/keyed_residual_scale_min": float(keyed_residual_scale.min().item()),
            "memory/keyed_residual_scale_max": float(keyed_residual_scale.max().item()),
            "memory/keyed_residual_target_norm": self.target_keyed_norm,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualFloorBoostPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        target_keyed_floor: float = 0.05,
        max_scale: float = 1.75,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.target_keyed_floor = float(target_keyed_floor)
        self.max_scale = float(max_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        keyed_norm = keyed_route_logits.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        keyed_residual_scale = (self.target_keyed_floor / keyed_norm).clamp(1.0, self.max_scale)
        boosted_keyed_route_logits = keyed_route_logits * keyed_residual_scale
        route_bias_logits = base_route_logits + boosted_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_norm.mean().item()),
            "memory/boosted_keyed_route_bias_logits_norm": float(boosted_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/keyed_residual_scale_mean": float(keyed_residual_scale.mean().item()),
            "memory/keyed_residual_scale_min": float(keyed_residual_scale.min().item()),
            "memory/keyed_residual_scale_max": float(keyed_residual_scale.max().item()),
            "memory/keyed_residual_target_floor": self.target_keyed_floor,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualRatioMatchedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        target_ratio: float = 0.1,
        min_ratio_scale: float = 0.5,
        max_ratio_scale: float = 3.0,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.target_ratio = float(target_ratio)
        self.min_ratio_scale = float(min_ratio_scale)
        self.max_ratio_scale = float(max_ratio_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        base_norm = base_route_logits.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        keyed_norm = keyed_route_logits.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        keyed_residual_scale = (self.target_ratio * base_norm / keyed_norm).clamp(self.min_ratio_scale, self.max_ratio_scale)
        scaled_keyed_route_logits = keyed_route_logits * keyed_residual_scale
        route_bias_logits = base_route_logits + scaled_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_norm.mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_norm.mean().item()),
            "memory/scaled_keyed_route_bias_logits_norm": float(scaled_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/keyed_residual_scale_mean": float(keyed_residual_scale.mean().item()),
            "memory/keyed_residual_scale_min": float(keyed_residual_scale.min().item()),
            "memory/keyed_residual_scale_max": float(keyed_residual_scale.max().item()),
            "memory/keyed_residual_target_ratio": self.target_ratio,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualMarginGatePhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        keyed_residual_boost: float = 0.5,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.keyed_residual_boost = float(keyed_residual_boost)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        top2_vals, _ = base_route_logits.topk(min(2, self.expert_count), dim=-1)
        if self.expert_count >= 2:
            top2_margin = top2_vals[:, 0] - top2_vals[:, 1]
        else:
            top2_margin = torch.zeros_like(top2_vals[:, 0])
        base_norm = base_route_logits.norm(dim=-1).clamp_min(1e-6)
        normalized_top2_margin = (top2_margin / (base_norm / math.sqrt(self.expert_count))).clamp(0.0, 1.0)
        keyed_gate = 1.0 + self.keyed_residual_boost * (1.0 - normalized_top2_margin)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        gated_keyed_route_logits = keyed_route_logits * keyed_gate.unsqueeze(-1)
        route_bias_logits = base_route_logits + gated_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_norm.mean().item()),
            "memory/base_route_top2_margin": float(top2_margin.mean().item()),
            "memory/base_route_top2_margin_normalized": float(normalized_top2_margin.mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/gated_keyed_route_bias_logits_norm": float(gated_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/keyed_residual_gate_mean": float(keyed_gate.mean().item()),
            "memory/keyed_residual_boost": self.keyed_residual_boost,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualMildBoostPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        keyed_residual_scale: float = 1.5,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.keyed_residual_scale = float(keyed_residual_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        boosted_keyed_route_logits = keyed_route_logits * self.keyed_residual_scale
        route_bias_logits = base_route_logits + boosted_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/boosted_keyed_route_bias_logits_norm": float(boosted_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/keyed_residual_scale": self.keyed_residual_scale,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualPerExpertScalePhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)
        self.keyed_residual_gates = nn.Parameter(torch.zeros(expert_count))

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        keyed_residual_scales = 2.0 * torch.sigmoid(self.keyed_residual_gates)
        scaled_keyed_route_logits = keyed_route_logits * keyed_residual_scales.unsqueeze(0)
        route_bias_logits = base_route_logits + scaled_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/scaled_keyed_route_bias_logits_norm": float(scaled_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/keyed_residual_scale_mean": float(keyed_residual_scales.mean().item()),
            "memory/keyed_residual_scale_std": float(keyed_residual_scales.std(unbiased=False).item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualLearnedScalePhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)
        self.keyed_residual_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        keyed_residual_scale = 2.0 * torch.sigmoid(self.keyed_residual_gate)
        scaled_keyed_route_logits = keyed_route_logits * keyed_residual_scale
        route_bias_logits = base_route_logits + scaled_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/scaled_keyed_route_bias_logits_norm": float(scaled_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/keyed_residual_scale": float(keyed_residual_scale.item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualTopKMaskedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        _, base_topk_idx = base_route_logits.topk(self.top_k, dim=-1)
        support_mask = torch.zeros_like(keyed_route_logits, dtype=torch.bool)
        support_mask.scatter_(1, base_topk_idx, True)
        masked_keyed_route_logits = keyed_route_logits.masked_fill(~support_mask, 0.0)
        route_bias_logits = base_route_logits + masked_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/masked_keyed_route_bias_logits_norm": float(masked_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/masked_keyed_route_bias_density": float(support_mask.float().mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualPostActivationPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        base_route_bias = torch.tanh(base_route_logits)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        keyed_route_residual = torch.tanh(keyed_route_logits)
        post_activation_route_bias = base_route_bias + keyed_route_residual
        route_bias = torch.tanh(post_activation_route_bias) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/base_route_bias_norm": float(base_route_bias.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_norm": float(keyed_route_residual.norm(dim=-1).mean().item()),
            "memory/post_activation_route_bias_norm": float(post_activation_route_bias.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualCenteredPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        centered_keyed_route_logits = keyed_route_logits - keyed_route_logits.mean(dim=-1, keepdim=True)
        route_bias_logits = base_route_logits + centered_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/centered_keyed_route_bias_logits_norm": float(centered_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/centered_keyed_route_bias_absmean": float(centered_keyed_route_logits.mean(dim=-1).abs().mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualBoostedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        keyed_residual_scale: float = 4.0,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.keyed_residual_scale = float(keyed_residual_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        boosted_keyed_route_logits = keyed_route_logits * self.keyed_residual_scale
        route_bias_logits = base_route_logits + boosted_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/boosted_keyed_route_bias_logits_norm": float(boosted_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/keyed_residual_scale": self.keyed_residual_scale,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualEntropyGatePhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        keyed_residual_base_scale: float = 0.5,
        keyed_residual_uncertainty_scale: float = 1.0,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.keyed_residual_base_scale = float(keyed_residual_base_scale)
        self.keyed_residual_uncertainty_scale = float(keyed_residual_uncertainty_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        base_route_probs = torch.softmax(base_route_logits / max(self.temperature, 1e-6), dim=-1)
        base_route_entropy = entropy_from_probs(base_route_probs)
        normalized_uncertainty = (base_route_entropy / math.log(self.expert_count)).clamp(0.0, 1.0)
        keyed_gate = self.keyed_residual_base_scale + self.keyed_residual_uncertainty_scale * normalized_uncertainty
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        gated_keyed_route_logits = keyed_route_logits * keyed_gate.unsqueeze(-1)
        route_bias_logits = base_route_logits + gated_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/base_route_bias_entropy": float(base_route_entropy.mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/gated_keyed_route_bias_logits_norm": float(gated_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/keyed_residual_gate_mean": float(keyed_gate.mean().item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualHiddenGatedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)
        self.keyed_gate_proj = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.keyed_gate_proj.weight)
        nn.init.zeros_(self.keyed_gate_proj.bias)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        keyed_gate = 0.5 + torch.sigmoid(self.keyed_gate_proj(hidden))
        gated_keyed_route_logits = keyed_route_logits * keyed_gate
        route_bias_logits = base_route_logits + gated_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/gated_keyed_route_bias_logits_norm": float(gated_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/keyed_residual_hidden_gate_mean": float(keyed_gate.mean().item()),
            "memory/keyed_residual_hidden_gate_std": float(keyed_gate.std(unbiased=False).item()),
            "memory/keyed_residual_hidden_gate_min": float(keyed_gate.min().item()),
            "memory/keyed_residual_hidden_gate_max": float(keyed_gate.max().item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualWeakBaseExpertGatedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        keyed_residual_weak_base_boost: float = 0.5,
        weak_base_factor: float = 1.5,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.keyed_residual_weak_base_boost = float(keyed_residual_weak_base_boost)
        self.weak_base_factor = float(weak_base_factor)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        base_abs = base_route_logits.abs()
        base_scale = base_abs.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        weak_base_gate = torch.clamp(1.0 - (base_abs / (base_scale * self.weak_base_factor)), min=0.0, max=1.0)
        keyed_residual_scale = 1.0 + weak_base_gate * self.keyed_residual_weak_base_boost
        gated_keyed_route_logits = keyed_route_logits * keyed_residual_scale
        route_bias_logits = base_route_logits + gated_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/gated_keyed_route_bias_logits_norm": float(gated_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/weak_base_expert_gate_mean": float(weak_base_gate.mean().item()),
            "memory/weak_base_expert_gate_std": float(weak_base_gate.std(unbiased=False).item()),
            "memory/weak_base_expert_gate_max": float(weak_base_gate.max().item()),
            "memory/keyed_residual_scale_mean": float(keyed_residual_scale.mean().item()),
            "memory/keyed_residual_weak_base_boost": self.keyed_residual_weak_base_boost,
            "memory/weak_base_factor": self.weak_base_factor,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualWeakBaseSignAlignedExpertGatedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        keyed_residual_weak_base_boost: float = 0.5,
        weak_base_factor: float = 1.5,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.keyed_residual_weak_base_boost = float(keyed_residual_weak_base_boost)
        self.weak_base_factor = float(weak_base_factor)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        base_abs = base_route_logits.abs()
        base_scale = base_abs.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        weak_base_gate = torch.clamp(1.0 - (base_abs / (base_scale * self.weak_base_factor)), min=0.0, max=1.0)
        sign_alignment_gate = (base_route_logits * keyed_route_logits > 0).to(dtype=base_route_logits.dtype)
        combined_gate = weak_base_gate * sign_alignment_gate
        keyed_residual_scale = 1.0 + combined_gate * self.keyed_residual_weak_base_boost
        gated_keyed_route_logits = keyed_route_logits * keyed_residual_scale
        route_bias_logits = base_route_logits + gated_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/gated_keyed_route_bias_logits_norm": float(gated_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/weak_base_expert_gate_mean": float(weak_base_gate.mean().item()),
            "memory/weak_base_expert_gate_std": float(weak_base_gate.std(unbiased=False).item()),
            "memory/sign_alignment_gate_mean": float(sign_alignment_gate.mean().item()),
            "memory/combined_sign_aligned_weak_base_gate_mean": float(combined_gate.mean().item()),
            "memory/combined_sign_aligned_weak_base_gate_max": float(combined_gate.max().item()),
            "memory/keyed_residual_scale_mean": float(keyed_residual_scale.mean().item()),
            "memory/keyed_residual_weak_base_boost": self.keyed_residual_weak_base_boost,
            "memory/weak_base_factor": self.weak_base_factor,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualWeakBaseKeyedTop2ExpertGatedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        keyed_residual_weak_base_boost: float = 0.5,
        weak_base_factor: float = 1.5,
        keyed_topk_count: int = 2,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.keyed_residual_weak_base_boost = float(keyed_residual_weak_base_boost)
        self.weak_base_factor = float(weak_base_factor)
        self.keyed_topk_count = int(keyed_topk_count)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        base_abs = base_route_logits.abs()
        base_scale = base_abs.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        weak_base_gate = torch.clamp(1.0 - (base_abs / (base_scale * self.weak_base_factor)), min=0.0, max=1.0)
        keyed_positive_gate = (keyed_route_logits > 0).to(dtype=keyed_route_logits.dtype)
        keyed_topk_count = max(1, min(self.keyed_topk_count, self.expert_count))
        keyed_topk_idx = keyed_route_logits.topk(k=keyed_topk_count, dim=-1).indices
        keyed_topk_gate = torch.zeros_like(keyed_route_logits)
        keyed_topk_gate.scatter_(dim=-1, index=keyed_topk_idx, value=1.0)
        combined_gate = weak_base_gate * keyed_positive_gate * keyed_topk_gate
        keyed_residual_scale = 1.0 + combined_gate * self.keyed_residual_weak_base_boost
        gated_keyed_route_logits = keyed_route_logits * keyed_residual_scale
        route_bias_logits = base_route_logits + gated_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/gated_keyed_route_bias_logits_norm": float(gated_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/weak_base_expert_gate_mean": float(weak_base_gate.mean().item()),
            "memory/weak_base_expert_gate_std": float(weak_base_gate.std(unbiased=False).item()),
            "memory/keyed_positive_gate_mean": float(keyed_positive_gate.mean().item()),
            "memory/keyed_top2_gate_mean": float(keyed_topk_gate.mean().item()),
            "memory/combined_weak_keyed_top2_gate_mean": float(combined_gate.mean().item()),
            "memory/combined_weak_keyed_top2_gate_max": float(combined_gate.max().item()),
            "memory/keyed_residual_scale_mean": float(keyed_residual_scale.mean().item()),
            "memory/keyed_residual_weak_base_boost": self.keyed_residual_weak_base_boost,
            "memory/weak_base_factor": self.weak_base_factor,
            "memory/keyed_top2_count": float(keyed_topk_count),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualWeakBaseKeyedAbsTop2ExpertGatedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        keyed_residual_weak_base_boost: float = 0.5,
        weak_base_factor: float = 1.5,
        keyed_topk_count: int = 2,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.keyed_residual_weak_base_boost = float(keyed_residual_weak_base_boost)
        self.weak_base_factor = float(weak_base_factor)
        self.keyed_topk_count = int(keyed_topk_count)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        base_abs = base_route_logits.abs()
        base_scale = base_abs.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        weak_base_gate = torch.clamp(1.0 - (base_abs / (base_scale * self.weak_base_factor)), min=0.0, max=1.0)
        keyed_abs = keyed_route_logits.abs()
        keyed_topk_count = max(1, min(self.keyed_topk_count, self.expert_count))
        keyed_topk_idx = keyed_abs.topk(k=keyed_topk_count, dim=-1).indices
        keyed_abs_topk_gate = torch.zeros_like(keyed_route_logits)
        keyed_abs_topk_gate.scatter_(dim=-1, index=keyed_topk_idx, value=1.0)
        combined_gate = weak_base_gate * keyed_abs_topk_gate
        keyed_residual_scale = 1.0 + combined_gate * self.keyed_residual_weak_base_boost
        gated_keyed_route_logits = keyed_route_logits * keyed_residual_scale
        route_bias_logits = base_route_logits + gated_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/gated_keyed_route_bias_logits_norm": float(gated_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/weak_base_expert_gate_mean": float(weak_base_gate.mean().item()),
            "memory/weak_base_expert_gate_std": float(weak_base_gate.std(unbiased=False).item()),
            "memory/keyed_abs_top2_gate_mean": float(keyed_abs_topk_gate.mean().item()),
            "memory/combined_weak_keyed_abs_top2_gate_mean": float(combined_gate.mean().item()),
            "memory/combined_weak_keyed_abs_top2_gate_max": float(combined_gate.max().item()),
            "memory/keyed_residual_scale_mean": float(keyed_residual_scale.mean().item()),
            "memory/keyed_residual_weak_base_boost": self.keyed_residual_weak_base_boost,
            "memory/weak_base_factor": self.weak_base_factor,
            "memory/keyed_abs_top2_count": float(keyed_topk_count),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualWeakBaseKeyedRelativeExpertGatedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        keyed_residual_weak_base_boost: float = 0.5,
        weak_base_factor: float = 1.5,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.keyed_residual_weak_base_boost = float(keyed_residual_weak_base_boost)
        self.weak_base_factor = float(weak_base_factor)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        base_abs = base_route_logits.abs()
        base_scale = base_abs.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        weak_base_gate = torch.clamp(1.0 - (base_abs / (base_scale * self.weak_base_factor)), min=0.0, max=1.0)
        keyed_abs = keyed_route_logits.abs()
        keyed_abs_max = keyed_abs.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        keyed_relative_gate = keyed_abs / keyed_abs_max
        combined_gate = weak_base_gate * keyed_relative_gate
        keyed_residual_scale = 1.0 + combined_gate * self.keyed_residual_weak_base_boost
        gated_keyed_route_logits = keyed_route_logits * keyed_residual_scale
        route_bias_logits = base_route_logits + gated_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/gated_keyed_route_bias_logits_norm": float(gated_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/weak_base_expert_gate_mean": float(weak_base_gate.mean().item()),
            "memory/weak_base_expert_gate_std": float(weak_base_gate.std(unbiased=False).item()),
            "memory/keyed_relative_gate_mean": float(keyed_relative_gate.mean().item()),
            "memory/keyed_relative_gate_std": float(keyed_relative_gate.std(unbiased=False).item()),
            "memory/keyed_relative_gate_max": float(keyed_relative_gate.max().item()),
            "memory/combined_weak_keyed_relative_gate_mean": float(combined_gate.mean().item()),
            "memory/combined_weak_keyed_relative_gate_max": float(combined_gate.max().item()),
            "memory/keyed_residual_scale_mean": float(keyed_residual_scale.mean().item()),
            "memory/keyed_residual_weak_base_boost": self.keyed_residual_weak_base_boost,
            "memory/weak_base_factor": self.weak_base_factor,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualWeakBaseExpertHighBaseGatedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        keyed_residual_weak_base_boost: float = 0.5,
        weak_base_factor: float = 1.5,
        high_base_target_norm: float = 0.7,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.keyed_residual_weak_base_boost = float(keyed_residual_weak_base_boost)
        self.weak_base_factor = float(weak_base_factor)
        self.high_base_target_norm = float(high_base_target_norm)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        base_abs = base_route_logits.abs()
        base_scale = base_abs.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        weak_base_gate = torch.clamp(1.0 - (base_abs / (base_scale * self.weak_base_factor)), min=0.0, max=1.0)
        base_route_logits_norm = base_route_logits.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        high_base_gate = torch.clamp((base_route_logits_norm / self.high_base_target_norm) - 1.0, min=0.0, max=1.0)
        combined_gate = weak_base_gate * high_base_gate
        keyed_residual_scale = 1.0 + combined_gate * self.keyed_residual_weak_base_boost
        gated_keyed_route_logits = keyed_route_logits * keyed_residual_scale
        route_bias_logits = base_route_logits + gated_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/gated_keyed_route_bias_logits_norm": float(gated_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/weak_base_expert_gate_mean": float(weak_base_gate.mean().item()),
            "memory/weak_base_expert_gate_std": float(weak_base_gate.std(unbiased=False).item()),
            "memory/high_base_gate_mean": float(high_base_gate.mean().item()),
            "memory/high_base_gate_max": float(high_base_gate.max().item()),
            "memory/combined_weak_high_base_gate_mean": float(combined_gate.mean().item()),
            "memory/combined_weak_high_base_gate_max": float(combined_gate.max().item()),
            "memory/keyed_residual_scale_mean": float(keyed_residual_scale.mean().item()),
            "memory/keyed_residual_weak_base_boost": self.keyed_residual_weak_base_boost,
            "memory/weak_base_factor": self.weak_base_factor,
            "memory/high_base_target_norm": self.high_base_target_norm,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualWeakBaseExpertHighBaseFloorGatedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        keyed_residual_weak_base_boost: float = 0.5,
        weak_base_factor: float = 1.5,
        high_base_target_norm: float = 0.7,
        low_base_boost_floor: float = 0.5,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.keyed_residual_weak_base_boost = float(keyed_residual_weak_base_boost)
        self.weak_base_factor = float(weak_base_factor)
        self.high_base_target_norm = float(high_base_target_norm)
        self.low_base_boost_floor = float(low_base_boost_floor)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        base_abs = base_route_logits.abs()
        base_scale = base_abs.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        weak_base_gate = torch.clamp(1.0 - (base_abs / (base_scale * self.weak_base_factor)), min=0.0, max=1.0)
        base_route_logits_norm = base_route_logits.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        high_base_gate = torch.clamp((base_route_logits_norm / self.high_base_target_norm) - 1.0, min=0.0, max=1.0)
        effective_high_base_floor_gate = self.low_base_boost_floor + (1.0 - self.low_base_boost_floor) * high_base_gate
        combined_gate = weak_base_gate * effective_high_base_floor_gate
        keyed_residual_scale = 1.0 + combined_gate * self.keyed_residual_weak_base_boost
        gated_keyed_route_logits = keyed_route_logits * keyed_residual_scale
        route_bias_logits = base_route_logits + gated_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/gated_keyed_route_bias_logits_norm": float(gated_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/weak_base_expert_gate_mean": float(weak_base_gate.mean().item()),
            "memory/weak_base_expert_gate_std": float(weak_base_gate.std(unbiased=False).item()),
            "memory/high_base_gate_mean": float(high_base_gate.mean().item()),
            "memory/high_base_gate_max": float(high_base_gate.max().item()),
            "memory/effective_high_base_floor_gate_mean": float(effective_high_base_floor_gate.mean().item()),
            "memory/effective_high_base_floor_gate_max": float(effective_high_base_floor_gate.max().item()),
            "memory/combined_weak_high_base_gate_mean": float(combined_gate.mean().item()),
            "memory/combined_weak_high_base_gate_max": float(combined_gate.max().item()),
            "memory/keyed_residual_scale_mean": float(keyed_residual_scale.mean().item()),
            "memory/keyed_residual_weak_base_boost": self.keyed_residual_weak_base_boost,
            "memory/weak_base_factor": self.weak_base_factor,
            "memory/high_base_target_norm": self.high_base_target_norm,
            "memory/low_base_boost_floor": self.low_base_boost_floor,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualWeakBaseExpertLowBaseRenormedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        keyed_residual_weak_base_boost: float = 0.5,
        weak_base_factor: float = 1.5,
        low_base_target_norm: float = 0.7,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.keyed_residual_weak_base_boost = float(keyed_residual_weak_base_boost)
        self.weak_base_factor = float(weak_base_factor)
        self.low_base_target_norm = float(low_base_target_norm)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        base_abs = base_route_logits.abs()
        base_scale = base_abs.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        weak_base_gate = torch.clamp(1.0 - (base_abs / (base_scale * self.weak_base_factor)), min=0.0, max=1.0)
        keyed_residual_scale = 1.0 + weak_base_gate * self.keyed_residual_weak_base_boost
        gated_keyed_route_logits = keyed_route_logits * keyed_residual_scale
        base_route_logits_norm = base_route_logits.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        low_base_renorm_gate = torch.clamp((self.low_base_target_norm / base_route_logits_norm) - 1.0, min=0.0, max=1.0)
        keyed_route_logits_norm = keyed_route_logits.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        gated_keyed_route_logits_norm = gated_keyed_route_logits.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        keyed_residual_norm_preserve_scale = keyed_route_logits_norm / gated_keyed_route_logits_norm
        renormed_keyed_route_logits = (gated_keyed_route_logits * keyed_residual_norm_preserve_scale).to(
            dtype=gated_keyed_route_logits.dtype
        )
        adaptive_keyed_route_logits = torch.lerp(
            gated_keyed_route_logits,
            renormed_keyed_route_logits,
            low_base_renorm_gate.to(dtype=gated_keyed_route_logits.dtype),
        )
        route_bias_logits = base_route_logits + adaptive_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/gated_keyed_route_bias_logits_norm": float(gated_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/renormed_keyed_route_bias_logits_norm": float(renormed_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/adaptive_keyed_route_bias_logits_norm": float(adaptive_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/weak_base_expert_gate_mean": float(weak_base_gate.mean().item()),
            "memory/weak_base_expert_gate_std": float(weak_base_gate.std(unbiased=False).item()),
            "memory/low_base_renorm_gate_mean": float(low_base_renorm_gate.mean().item()),
            "memory/low_base_renorm_gate_max": float(low_base_renorm_gate.max().item()),
            "memory/keyed_residual_scale_mean": float(keyed_residual_scale.mean().item()),
            "memory/keyed_residual_norm_preserve_scale_mean": float(keyed_residual_norm_preserve_scale.mean().item()),
            "memory/keyed_residual_weak_base_boost": self.keyed_residual_weak_base_boost,
            "memory/weak_base_factor": self.weak_base_factor,
            "memory/low_base_target_norm": self.low_base_target_norm,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualWeakBaseExpertRenormedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        keyed_residual_weak_base_boost: float = 0.5,
        weak_base_factor: float = 1.5,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.keyed_residual_weak_base_boost = float(keyed_residual_weak_base_boost)
        self.weak_base_factor = float(weak_base_factor)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        base_abs = base_route_logits.abs()
        base_scale = base_abs.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        weak_base_gate = torch.clamp(1.0 - (base_abs / (base_scale * self.weak_base_factor)), min=0.0, max=1.0)
        keyed_residual_scale = 1.0 + weak_base_gate * self.keyed_residual_weak_base_boost
        reweighted_keyed_route_logits = keyed_route_logits * keyed_residual_scale
        keyed_route_logits_norm = keyed_route_logits.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        reweighted_keyed_route_logits_norm = reweighted_keyed_route_logits.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        keyed_residual_norm_preserve_scale = keyed_route_logits_norm / reweighted_keyed_route_logits_norm
        renormed_keyed_route_logits = reweighted_keyed_route_logits * keyed_residual_norm_preserve_scale
        route_bias_logits = base_route_logits + renormed_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/reweighted_keyed_route_bias_logits_norm": float(reweighted_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/renormed_keyed_route_bias_logits_norm": float(renormed_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/weak_base_expert_gate_mean": float(weak_base_gate.mean().item()),
            "memory/weak_base_expert_gate_std": float(weak_base_gate.std(unbiased=False).item()),
            "memory/weak_base_expert_gate_max": float(weak_base_gate.max().item()),
            "memory/keyed_residual_scale_mean": float(keyed_residual_scale.mean().item()),
            "memory/keyed_residual_norm_preserve_scale_mean": float(keyed_residual_norm_preserve_scale.mean().item()),
            "memory/keyed_residual_weak_base_boost": self.keyed_residual_weak_base_boost,
            "memory/weak_base_factor": self.weak_base_factor,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedKeyedResidualWeakBaseWeakKeyedExpertGatedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
        keyed_residual_weak_base_boost: float = 0.5,
        weak_base_factor: float = 1.5,
        weak_keyed_factor: float = 1.5,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.keyed_residual_weak_base_boost = float(keyed_residual_weak_base_boost)
        self.weak_base_factor = float(weak_base_factor)
        self.weak_keyed_factor = float(weak_keyed_factor)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_logits = self.route_bias_proj(hidden)
        route_memory_query = self.query(hidden)
        keyed_route_logits = route_memory_query @ self.expert_keys.t() / math.sqrt(route_memory_query.size(-1))
        base_abs = base_route_logits.abs()
        keyed_abs = keyed_route_logits.abs()
        base_scale = base_abs.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        keyed_scale = keyed_abs.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        weak_base_gate = torch.clamp(1.0 - (base_abs / (base_scale * self.weak_base_factor)), min=0.0, max=1.0)
        weak_keyed_gate = torch.clamp(1.0 - (keyed_abs / (keyed_scale * self.weak_keyed_factor)), min=0.0, max=1.0)
        combined_gate = weak_base_gate * weak_keyed_gate
        keyed_residual_scale = 1.0 + combined_gate * self.keyed_residual_weak_base_boost
        gated_keyed_route_logits = keyed_route_logits * keyed_residual_scale
        route_bias_logits = base_route_logits + gated_keyed_route_logits
        route_bias = torch.tanh(route_bias_logits) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_logits_norm": float(base_route_logits.norm(dim=-1).mean().item()),
            "memory/keyed_route_bias_logits_norm": float(keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/gated_keyed_route_bias_logits_norm": float(gated_keyed_route_logits.norm(dim=-1).mean().item()),
            "memory/route_bias_logits_norm": float(route_bias_logits.norm(dim=-1).mean().item()),
            "memory/route_memory_query_norm": float(route_memory_query.norm(dim=-1).mean().item()),
            "memory/weak_base_expert_gate_mean": float(weak_base_gate.mean().item()),
            "memory/weak_base_expert_gate_std": float(weak_base_gate.std(unbiased=False).item()),
            "memory/weak_keyed_expert_gate_mean": float(weak_keyed_gate.mean().item()),
            "memory/weak_keyed_expert_gate_std": float(weak_keyed_gate.std(unbiased=False).item()),
            "memory/combined_weak_expert_gate_mean": float(combined_gate.mean().item()),
            "memory/combined_weak_expert_gate_max": float(combined_gate.max().item()),
            "memory/keyed_residual_scale_mean": float(keyed_residual_scale.mean().item()),
            "memory/keyed_residual_weak_base_boost": self.keyed_residual_weak_base_boost,
            "memory/weak_base_factor": self.weak_base_factor,
            "memory/weak_keyed_factor": self.weak_keyed_factor,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedContextualPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)
        self.route_context_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        token_summary = masked_mean(tokens)
        hidden = self._prepare_hidden(
            state,
            token_summary,
            done,
        )
        route_memory_bias = self.route_bias_proj(hidden)
        route_context_bias = self.route_context_proj(token_summary)
        route_bias = torch.tanh(route_memory_bias + route_context_bias) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/route_context_bias_norm": float(route_context_bias.norm(dim=-1).mean().item()),
            "memory/route_memory_bias_norm": float(route_memory_bias.norm(dim=-1).mean().item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedScaleGatedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)
        self.route_scale_gate = nn.Linear(hidden_size, 1)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        token_summary = masked_mean(tokens)
        hidden = self._prepare_hidden(
            state,
            token_summary,
            done,
        )
        base_route_bias = torch.tanh(self.route_bias_proj(hidden))
        scale_gate = torch.sigmoid(self.route_scale_gate(token_summary))
        route_bias = base_route_bias * (self.route_memory_scale * scale_gate)
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/route_scale_gate_mean": float(scale_gate.mean().item()),
            "memory/base_route_bias_norm": float(base_route_bias.norm(dim=-1).mean().item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedNormalizedPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        base_route_bias = torch.tanh(self.route_bias_proj(hidden))
        base_norm = base_route_bias.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        route_bias = (base_route_bias / base_norm) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/base_route_bias_norm": float(base_route_bias.norm(dim=-1).mean().item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedLayerNormPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_hidden_norm = nn.LayerNorm(hidden_size)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        norm_hidden = self.route_hidden_norm(hidden)
        route_bias = torch.tanh(self.route_bias_proj(norm_hidden)) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/route_hidden_prenorm_norm": float(hidden.norm(dim=-1).mean().item()),
            "memory/route_hidden_postnorm_norm": float(norm_hidden.norm(dim=-1).mean().item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedGatedPhaseMemoryCore(RoutedExpertGatedPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        memory_gate_bias: float,
        memory_reset_bias: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
            memory_gate_bias=memory_gate_bias,
            memory_reset_bias=memory_reset_bias,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        hidden = self._prepare_hidden(
            state,
            masked_mean(tokens),
            done,
        )
        route_bias = torch.tanh(self.route_bias_proj(hidden)) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, hidden, route_probs)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )


class RoutedExpertRouteBiasedDualPhaseMemoryCore(RoutedExpertPhaseMemoryCore):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        top_k: int,
        temperature: float,
        memory_mix: float,
        route_memory_scale: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
            memory_mix=memory_mix,
        )
        self.route_memory_scale = float(route_memory_scale)
        self.route_bias_proj = nn.Linear(hidden_size, expert_count)
        self.route_memory_cell = nn.GRUCell(hidden_size, hidden_size)

    def initial_state(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        zeros = torch.zeros(batch_size, self.hidden_size, device=device)
        return {"hidden": zeros.clone(), "route_hidden": zeros}

    def _prepare_named_hidden(
        self,
        state: dict[str, torch.Tensor],
        key: str,
        pooled: torch.Tensor,
        done: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden = state.get(key)
        if hidden is None:
            hidden = torch.zeros(pooled.size(0), self.hidden_size, device=pooled.device, dtype=pooled.dtype)
        if done is not None:
            hidden = hidden * (~done).to(dtype=pooled.dtype).unsqueeze(-1)
        return hidden

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        token_summary = masked_mean(tokens)
        route_hidden = self._prepare_named_hidden(state, "route_hidden", token_summary, done)
        route_bias = torch.tanh(self.route_bias_proj(route_hidden)) * self.route_memory_scale
        route_probs, topk_values, topk_idx = self.route(tokens, route_bias=route_bias)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)
        phase_hidden = self._prepare_named_hidden(state, "hidden", pooled, done)
        pooled, next_hidden, memory_metrics = self._apply_memory(pooled, phase_hidden, route_probs)
        next_route_hidden = self.route_memory_cell(pooled, route_hidden)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            **memory_metrics,
            "memory/route_bias_norm": float(route_bias.norm(dim=-1).mean().item()),
            "memory/route_bias_absmax": float(route_bias.abs().max().item()),
            "memory/route_bias_scale": self.route_memory_scale,
            "memory/route_hidden_norm": float(next_route_hidden.norm(dim=-1).mean().item()),
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={
                "hidden": next_hidden.detach(),
                "route_hidden": next_route_hidden.detach(),
            },
        )
