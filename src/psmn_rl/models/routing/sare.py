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

    def route(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.query(tokens)
        scores = query @ self.expert_keys.t() / math.sqrt(query.size(-1))
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

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        route_probs, topk_values, topk_idx = self.route(tokens)
        mixed = self.apply_experts(tokens, topk_values, topk_idx)
        tokens = self.output_norm(tokens + mixed)
        pooled = masked_mean(tokens)

        hidden = state.get("hidden")
        if hidden is None:
            hidden = torch.zeros(pooled.size(0), self.hidden_size, device=pooled.device, dtype=pooled.dtype)
        if done is not None:
            hidden = hidden * (~done).float().unsqueeze(-1)

        next_hidden = self.memory_cell(pooled, hidden)
        pooled = torch.lerp(pooled, next_hidden, self.memory_mix)
        metrics = {
            **_route_metrics(route_probs, topk_idx, self.expert_count),
            **token_representation_metrics(tokens, pooled),
            "memory/hidden_norm": float(next_hidden.norm(dim=-1).mean().item()),
            "memory/mix": self.memory_mix,
        }
        return CoreOutput(
            pooled=pooled,
            tokens=tokens,
            metrics=metrics,
            next_state={"hidden": next_hidden.detach()},
        )
