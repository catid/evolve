from __future__ import annotations

import math

import torch
from torch import nn

from psmn_rl.metrics import reduce_path_statistics
from psmn_rl.models.common import CoreOutput, entropy_from_probs, masked_mean
from psmn_rl.models.cores.experts import ExpertBank
from psmn_rl.models.encoders import build_token_encoder


class TREGHCore(nn.Module):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        expert_count: int,
        expert_hidden_size: int,
        temperature: float,
        ponder_cost: float,
        max_hops: int,
        halt_bias: float,
    ) -> None:
        super().__init__()
        self.expert_count = expert_count
        self.temperature = temperature
        self.ponder_cost = ponder_cost
        self.max_hops = max(1, min(int(max_hops), 2))
        self.halt_bias = halt_bias
        self.encoder = build_token_encoder(observation_space, token_dim, patch_size)
        self.input_proj = nn.Linear(token_dim, hidden_size)
        self.query_1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query_2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.keys = nn.Parameter(torch.randn(expert_count, hidden_size) / math.sqrt(hidden_size))
        self.transition_bias = nn.Parameter(torch.zeros(expert_count, expert_count))
        self.halt_head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        self.bank = ExpertBank(hidden_size, expert_hidden_size, expert_count)
        self.norm = nn.LayerNorm(hidden_size)

    def _dispatch(self, tokens: torch.Tensor, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(scores / self.temperature, dim=-1)
        top1_idx = probs.argmax(dim=-1)
        outputs = self.bank.forward_all(tokens)
        mixed = (outputs * probs.unsqueeze(-1)).sum(dim=2)
        return mixed, top1_idx

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        scores_1 = self.query_1(tokens) @ self.keys.t() / math.sqrt(tokens.size(-1))
        hop_1, top1_idx = self._dispatch(tokens, scores_1)
        halt_prob = torch.sigmoid(self.halt_head(hop_1) + self.halt_bias)

        if self.max_hops > 1:
            scores_2 = self.query_2(hop_1) @ self.keys.t() / math.sqrt(tokens.size(-1))
            scores_2 = scores_2 + self.transition_bias[top1_idx]
            route_probs_2 = torch.softmax(scores_2 / self.temperature, dim=-1)
            hop_2 = (self.bank.forward_all(hop_1) * route_probs_2.unsqueeze(-1)).sum(dim=2)
            continue_prob = 1.0 - halt_prob
            effective_halt = halt_prob
        else:
            route_probs_2 = torch.zeros_like(scores_1.softmax(dim=-1))
            hop_2 = hop_1
            continue_prob = torch.zeros_like(halt_prob)
            effective_halt = torch.ones_like(halt_prob)

        mixed = effective_halt * hop_1 + continue_prob * hop_2
        tokens = self.norm(tokens + mixed)
        pooled = masked_mean(tokens)

        pair_ids = top1_idx * self.expert_count + route_probs_2.argmax(dim=-1)
        path_counts = torch.bincount(pair_ids.reshape(-1), minlength=self.expert_count * self.expert_count)
        metrics = {
            "route_entropy": float(entropy_from_probs(torch.softmax(scores_1, dim=-1)).mean().item()),
            "path_entropy": reduce_path_statistics(path_counts)["path_entropy"],
            "path_coverage_top_10": reduce_path_statistics(path_counts)["path_coverage_top_10"],
            "path_coverage_top_20": reduce_path_statistics(path_counts)["path_coverage_top_20"],
            "path_coverage_top_50": reduce_path_statistics(path_counts)["path_coverage_top_50"],
            "avg_hop_count": float((1.0 + continue_prob).mean().item()),
            "avg_halting_probability": float(effective_halt.mean().item()),
            "active_compute_proxy": float(1.0 + continue_prob.mean().item()),
            "max_hops": float(self.max_hops),
            "halt_bias": float(self.halt_bias),
        }
        avg_usage = 0.5 * (
            torch.softmax(scores_1, dim=-1).mean(dim=(0, 1)) + route_probs_2.mean(dim=(0, 1))
        )
        for expert_index, value in enumerate(avg_usage):
            metrics[f"expert_load_{expert_index}"] = float(value.item())
        aux_losses = {"ponder_loss": continue_prob.mean() * self.ponder_cost}
        return CoreOutput(pooled=pooled, tokens=tokens, metrics=metrics, next_state={}, aux_losses=aux_losses)
