from __future__ import annotations

import gymnasium as gym
import torch
from torch import nn


class MiniGridTokenEncoder(nn.Module):
    def __init__(self, observation_space: gym.Space, token_dim: int) -> None:
        super().__init__()
        image_space = observation_space["image"]
        height, width, _ = image_space.shape
        self.height = height
        self.width = width
        self.max_tokens = height * width + 1
        self.object_embed = nn.Embedding(16, token_dim)
        self.color_embed = nn.Embedding(16, token_dim)
        self.state_embed = nn.Embedding(16, token_dim)
        self.direction_embed = nn.Embedding(4, token_dim)
        self.position_embed = nn.Embedding(self.max_tokens, token_dim)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        image = obs["image"].long()
        batch_size = image.shape[0]
        height, width = image.shape[1:3]
        tokens = (
            self.object_embed(image[..., 0])
            + self.color_embed(image[..., 1])
            + self.state_embed(image[..., 2])
        )
        tokens = tokens.view(batch_size, height * width, -1)
        positions = torch.arange(height * width, device=image.device)
        tokens = tokens + self.position_embed(positions).unsqueeze(0)
        direction = obs["direction"].long().clamp_min(0).clamp_max(3)
        direction_token = self.direction_embed(direction) + self.position_embed(
            torch.full((batch_size,), height * width, device=image.device, dtype=torch.long)
        )
        direction_token = direction_token.unsqueeze(1)
        return self.norm(torch.cat([tokens, direction_token], dim=1))


class MiniGridFlatEncoder(nn.Module):
    def __init__(self, observation_space: gym.Space, hidden_size: int) -> None:
        super().__init__()
        image_shape = observation_space["image"].shape
        flat_dim = image_shape[0] * image_shape[1] * image_shape[2] + 4
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        image = obs["image"].float()
        batch_size = image.shape[0]
        flat_image = image.view(batch_size, -1) / 10.0
        direction = torch.nn.functional.one_hot(obs["direction"].long().clamp_min(0).clamp_max(3), num_classes=4).float()
        return self.mlp(torch.cat([flat_image, direction], dim=-1))
