from __future__ import annotations

import torch
from torch import nn

from psmn_rl.models.common import CoreOutput, masked_mean
from psmn_rl.models.encoders import build_flat_encoder, build_token_encoder


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm_1(tokens)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        tokens = tokens + attn_out
        return tokens + self.mlp(self.norm_2(tokens))


class FlatDenseCore(nn.Module):
    def __init__(self, observation_space, hidden_size: int) -> None:
        super().__init__()
        self.encoder = build_flat_encoder(observation_space, hidden_size)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        pooled = self.encoder(obs)
        metrics = {"route_entropy": 0.0, "active_compute_proxy": 1.0}
        return CoreOutput(pooled=pooled, tokens=None, metrics=metrics, next_state={})


class TokenDenseCore(nn.Module):
    def __init__(
        self,
        observation_space,
        token_dim: int,
        patch_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = build_token_encoder(observation_space, token_dim, patch_size)
        self.input_proj = nn.Linear(token_dim, hidden_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        pooled = masked_mean(tokens)
        metrics = {"route_entropy": 0.0, "active_compute_proxy": 1.0}
        return CoreOutput(pooled=pooled, tokens=tokens, metrics=metrics, next_state={})
