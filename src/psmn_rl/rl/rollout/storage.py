from __future__ import annotations

from dataclasses import dataclass

import torch


TensorDict = dict[str, torch.Tensor]


def stack_tensor_dict(items: list[TensorDict]) -> TensorDict:
    if not items:
        return {}
    keys = items[0].keys()
    return {key: torch.stack([item[key] for item in items], dim=0) for key in keys}


def flatten_tensor_dict(items: TensorDict) -> TensorDict:
    return {key: value.flatten(0, 1) for key, value in items.items()}


@dataclass
class RolloutBatch:
    obs: TensorDict
    states: TensorDict
    done_inputs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    next_dones: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def flatten(self) -> "RolloutBatch":
        return RolloutBatch(
            obs=flatten_tensor_dict(self.obs),
            states=flatten_tensor_dict(self.states),
            done_inputs=self.done_inputs.flatten(0, 1),
            actions=self.actions.flatten(0, 1),
            log_probs=self.log_probs.flatten(0, 1),
            values=self.values.flatten(0, 1),
            rewards=self.rewards.flatten(0, 1),
            next_dones=self.next_dones.flatten(0, 1),
            advantages=self.advantages.flatten(0, 1),
            returns=self.returns.flatten(0, 1),
        )


class RolloutStorage:
    def __init__(self) -> None:
        self.obs: list[TensorDict] = []
        self.states: list[TensorDict] = []
        self.done_inputs: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.next_dones: list[torch.Tensor] = []

    def add(
        self,
        obs: TensorDict,
        state: TensorDict,
        done_input: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        next_done: torch.Tensor,
    ) -> None:
        self.obs.append({key: value.detach() for key, value in obs.items()})
        self.states.append({key: value.detach() for key, value in state.items()})
        self.done_inputs.append(done_input.detach())
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.rewards.append(reward.detach())
        self.next_dones.append(next_done.detach())

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        last_done: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> RolloutBatch:
        rewards = torch.stack(self.rewards)
        values = torch.stack(self.values)
        next_dones = torch.stack(self.next_dones)
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros_like(last_value)
        for step in reversed(range(rewards.size(0))):
            if step == rewards.size(0) - 1:
                next_non_terminal = 1.0 - last_done.float()
                next_value = last_value
            else:
                next_non_terminal = 1.0 - next_dones[step].float()
                next_value = values[step + 1]
            delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
            last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
            advantages[step] = last_advantage
        returns = advantages + values
        return RolloutBatch(
            obs=stack_tensor_dict(self.obs),
            states=stack_tensor_dict(self.states),
            done_inputs=torch.stack(self.done_inputs),
            actions=torch.stack(self.actions),
            log_probs=torch.stack(self.log_probs),
            values=values,
            rewards=rewards,
            next_dones=next_dones,
            advantages=advantages,
            returns=returns,
        )
