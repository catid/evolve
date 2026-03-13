from __future__ import annotations

import torch
from torch import nn

from psmn_rl.models.common import CoreOutput, masked_mean
from psmn_rl.models.routing.sare import RoutedExpertCore


class SRWCore(nn.Module):
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
        num_heads: int,
        relational_tokens: int,
    ) -> None:
        super().__init__()
        self.unary = RoutedExpertCore(
            observation_space=observation_space,
            token_dim=token_dim,
            patch_size=patch_size,
            hidden_size=hidden_size,
            expert_count=expert_count,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            temperature=temperature,
        )
        self.relational_tokens = relational_tokens
        self.gate = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        self.salience = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        self.rel_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.rel_norm = nn.LayerNorm(hidden_size)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        unary_out = self.unary(obs, state, done)
        assert unary_out.tokens is not None
        tokens = unary_out.tokens
        pooled = masked_mean(tokens)
        gate_prob = torch.sigmoid(self.gate(pooled)).unsqueeze(1)
        salience = self.salience(tokens).squeeze(-1)
        top_m = min(self.relational_tokens, tokens.size(1))
        salient_idx = salience.topk(k=top_m, dim=-1).indices
        gather_index = salient_idx.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
        selected = torch.gather(tokens, 1, gather_index)
        rel_out, _ = self.rel_attn(selected, selected, selected, need_weights=False)
        rel_out = self.rel_norm(selected + rel_out)
        scatter = torch.zeros_like(tokens)
        scatter.scatter_add_(1, gather_index, rel_out)
        tokens = tokens + gate_prob * scatter
        pooled = masked_mean(tokens)
        metrics = dict(unary_out.metrics)
        metrics["relational_usage_rate"] = float(gate_prob.mean().item())
        metrics["active_compute_proxy"] = float(metrics["active_compute_proxy"] + gate_prob.mean().item())
        return CoreOutput(pooled=pooled, tokens=tokens, metrics=metrics, next_state={})
