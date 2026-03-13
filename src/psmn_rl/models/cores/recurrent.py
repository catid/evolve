from __future__ import annotations

import torch
from torch import nn

from psmn_rl.models.common import CoreOutput, masked_mean
from psmn_rl.models.cores.dense import TransformerBlock
from psmn_rl.models.encoders import build_token_encoder


class TokenGRUCore(nn.Module):
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
        self.hidden_size = hidden_size
        self.encoder = build_token_encoder(observation_space, token_dim, patch_size)
        self.input_proj = nn.Linear(token_dim, hidden_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.gru = nn.GRUCell(hidden_size, hidden_size)

    def initial_state(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        return {"hidden": torch.zeros(batch_size, self.hidden_size, device=device)}

    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done: torch.Tensor | None) -> CoreOutput:
        tokens = self.input_proj(self.encoder(obs))
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        pooled = masked_mean(tokens)

        hidden = state.get("hidden")
        if hidden is None:
            hidden = torch.zeros(pooled.size(0), self.hidden_size, device=pooled.device, dtype=pooled.dtype)
        if done is not None:
            hidden = hidden * (~done).float().unsqueeze(-1)

        next_hidden = self.gru(pooled, hidden)
        metrics = {"route_entropy": 0.0, "active_compute_proxy": 1.0}
        return CoreOutput(pooled=next_hidden, tokens=tokens, metrics=metrics, next_state={"hidden": next_hidden.detach()})
