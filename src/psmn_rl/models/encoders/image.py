from __future__ import annotations

import gymnasium as gym
import torch
from torch import nn


class ImagePatchTokenEncoder(nn.Module):
    def __init__(self, observation_space: gym.Space, token_dim: int, patch_size: int) -> None:
        super().__init__()
        height, width, channels = observation_space.shape
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("Image patch encoder requires observation shape divisible by patch_size")
        self.patch_proj = nn.Conv2d(channels, token_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (height // patch_size) * (width // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.position_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, token_dim))
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        pixels = obs["pixels"].float().permute(0, 3, 1, 2) / 255.0
        patches = self.patch_proj(pixels).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(patches.size(0), -1, -1)
        tokens = torch.cat([cls, patches], dim=1) + self.position_embed[:, : patches.size(1) + 1]
        return self.norm(tokens)


class ImageFlatEncoder(nn.Module):
    def __init__(self, observation_space: gym.Space, hidden_size: int) -> None:
        super().__init__()
        height, width, channels = observation_space.shape
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.GELU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            conv_dim = self.conv(dummy).flatten(1).size(1)
        self.proj = nn.Sequential(
            nn.Linear(conv_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        pixels = obs["pixels"].float().permute(0, 3, 1, 2) / 255.0
        return self.proj(self.conv(pixels).flatten(1))
