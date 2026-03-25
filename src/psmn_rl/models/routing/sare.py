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
