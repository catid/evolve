from __future__ import annotations

import gymnasium as gym


class EpisodeSuccessWrapper(gym.Wrapper):
    """Expose a durable per-step success boolean for episode summaries."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        if "success" not in info:
            info["success"] = bool(terminated and reward > 0.0)
        return obs, reward, terminated, truncated, info
