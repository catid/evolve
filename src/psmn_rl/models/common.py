from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


TensorDict = dict[str, torch.Tensor]


@dataclass
class CoreOutput:
    pooled: torch.Tensor
    tokens: torch.Tensor | None
    metrics: dict[str, torch.Tensor | float] = field(default_factory=dict)
    next_state: TensorDict = field(default_factory=dict)
    aux_losses: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class ModelOutput:
    logits: torch.Tensor
    value: torch.Tensor
    metrics: dict[str, torch.Tensor | float]
    next_state: TensorDict
    aux_losses: dict[str, torch.Tensor]


def masked_mean(tokens: torch.Tensor) -> torch.Tensor:
    return tokens.mean(dim=1)


def entropy_from_probs(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=dim)


def pooled_representation_metrics(pooled: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"repr/pooled_norm": pooled.norm(dim=-1)}


def token_representation_metrics(tokens: torch.Tensor, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
    spatial_tokens = tokens[:, :-1] if tokens.size(1) > 1 else tokens
    direction_token = tokens[:, -1]
    metrics: dict[str, torch.Tensor] = {
        "repr/pooled_norm": pooled.norm(dim=-1),
        "repr/token_norm": spatial_tokens.norm(dim=-1).mean(dim=-1),
        "repr/token_feature_std": spatial_tokens.std(dim=1, unbiased=False).mean(dim=-1),
        "repr/direction_norm": direction_token.norm(dim=-1),
    }
    if spatial_tokens.size(1) > 1:
        normalized = F.normalize(spatial_tokens, dim=-1)
        cosine = normalized @ normalized.transpose(1, 2)
        token_count = spatial_tokens.size(1)
        off_diag = (cosine.sum(dim=(1, 2)) - token_count) / max(token_count * (token_count - 1), 1)
        metrics["repr/token_pairwise_cosine"] = off_diag
    else:
        metrics["repr/token_pairwise_cosine"] = torch.zeros(
            spatial_tokens.size(0), device=tokens.device, dtype=tokens.dtype
        )
    return metrics


class ActorCriticModel(nn.Module):
    def __init__(
        self,
        core: nn.Module,
        action_dim: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.core = core
        self.policy_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> TensorDict:
        if hasattr(self.core, "initial_state"):
            return getattr(self.core, "initial_state")(batch_size, device)
        return {}

    def forward(self, obs: TensorDict, state: TensorDict | None = None, done: torch.Tensor | None = None) -> ModelOutput:
        core_output: CoreOutput = self.core(obs, state or {}, done)
        logits = self.policy_head(core_output.pooled)
        value = self.value_head(core_output.pooled).squeeze(-1)
        return ModelOutput(
            logits=logits,
            value=value,
            metrics=core_output.metrics,
            next_state=core_output.next_state,
            aux_losses=core_output.aux_losses,
        )

    def get_dist(self, logits: torch.Tensor, temperature: float = 1.0) -> Categorical:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        return Categorical(logits=logits / temperature)

    def act(
        self,
        obs: TensorDict,
        state: TensorDict | None = None,
        done: torch.Tensor | None = None,
        greedy: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TensorDict, dict[str, torch.Tensor | float], dict[str, torch.Tensor]]:
        output = self.forward(obs, state=state, done=done)
        dist = self.get_dist(output.logits, temperature=temperature)
        if greedy:
            action = output.logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, output.value, output.next_state, output.metrics, output.aux_losses

    def evaluate_actions(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        state: TensorDict | None = None,
        done: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        output = self.forward(obs, state=state, done=done)
        dist = self.get_dist(output.logits)
        return {
            "log_prob": dist.log_prob(actions),
            "entropy": dist.entropy(),
            "value": output.value,
            "metrics": output.metrics,
            "next_state": output.next_state,
            "aux_losses": output.aux_losses,
        }
