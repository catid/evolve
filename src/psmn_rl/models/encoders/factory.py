from __future__ import annotations

import gymnasium as gym

from psmn_rl.models.encoders.image import ImageFlatEncoder, ImagePatchTokenEncoder
from psmn_rl.models.encoders.minigrid import MiniGridFlatEncoder, MiniGridTokenEncoder


def build_token_encoder(observation_space: gym.Space, token_dim: int, patch_size: int):
    if isinstance(observation_space, gym.spaces.Dict):
        return MiniGridTokenEncoder(observation_space, token_dim)
    if isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 3:
        return ImagePatchTokenEncoder(observation_space, token_dim, patch_size)
    raise ValueError(f"Unsupported observation space for token encoder: {observation_space}")


def build_flat_encoder(observation_space: gym.Space, hidden_size: int):
    if isinstance(observation_space, gym.spaces.Dict):
        return MiniGridFlatEncoder(observation_space, hidden_size)
    if isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 3:
        return ImageFlatEncoder(observation_space, hidden_size)
    raise ValueError(f"Unsupported observation space for flat encoder: {observation_space}")
