from __future__ import annotations

import pytest
import torch

from psmn_rl.rl.rollout.storage import RolloutStorage


def _single_step_obs(value: float) -> dict[str, torch.Tensor]:
    return {"pixels": torch.tensor([[value]], dtype=torch.float32)}


def test_compute_returns_and_advantages_matches_exact_gae() -> None:
    storage = RolloutStorage()
    storage.add(
        obs=_single_step_obs(1.0),
        state={},
        done_input=torch.tensor([True]),
        action=torch.tensor([0]),
        log_prob=torch.tensor([0.0]),
        value=torch.tensor([0.5]),
        reward=torch.tensor([1.0]),
        next_done=torch.tensor([False]),
    )
    storage.add(
        obs=_single_step_obs(2.0),
        state={},
        done_input=torch.tensor([False]),
        action=torch.tensor([0]),
        log_prob=torch.tensor([0.0]),
        value=torch.tensor([0.25]),
        reward=torch.tensor([2.0]),
        next_done=torch.tensor([True]),
    )

    batch = storage.compute_returns_and_advantages(
        last_value=torch.tensor([0.0]),
        last_done=torch.tensor([True]),
        gamma=0.9,
        gae_lambda=1.0,
    )

    assert batch.advantages[:, 0].tolist() == pytest.approx([2.3, 1.75])
    assert batch.returns[:, 0].tolist() == pytest.approx([2.8, 2.0])


def test_truncated_transition_is_correct_once_reward_is_bootstrapped() -> None:
    storage = RolloutStorage()
    storage.add(
        obs=_single_step_obs(1.0),
        state={},
        done_input=torch.tensor([True]),
        action=torch.tensor([0]),
        log_prob=torch.tensor([0.0]),
        value=torch.tensor([0.5]),
        reward=torch.tensor([4.6]),
        next_done=torch.tensor([True]),
    )

    batch = storage.compute_returns_and_advantages(
        last_value=torch.tensor([0.0]),
        last_done=torch.tensor([True]),
        gamma=0.9,
        gae_lambda=1.0,
    )

    assert batch.advantages[:, 0].tolist() == pytest.approx([4.1])
    assert batch.returns[:, 0].tolist() == pytest.approx([4.6])
