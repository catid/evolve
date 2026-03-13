from __future__ import annotations

from torch import nn


class ExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, expert_hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, expert_hidden_size),
            nn.GELU(),
            nn.Linear(expert_hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.net(x)


class ExpertBank(nn.Module):
    def __init__(self, hidden_size: int, expert_hidden_size: int, expert_count: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [ExpertMLP(hidden_size, expert_hidden_size) for _ in range(expert_count)]
        )

    def forward_all(self, tokens):
        import torch

        return torch.stack([expert(tokens) for expert in self.experts], dim=2)
