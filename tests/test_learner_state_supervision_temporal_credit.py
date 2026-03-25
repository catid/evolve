from __future__ import annotations

import torch

from psmn_rl.analysis.learner_state_supervision import (
    EvalConfig,
    LearnerStateSpec,
    LoopConfig,
    StudentConfig,
    TeacherConfig,
    _apply_weight_adjustments,
    _warmup_only_round,
)
from psmn_rl.analysis.policy_distillation import DistillationBatch, _dataset_weights


def _batch(*, steps_from_end: list[int], disagreement: list[float] | None = None) -> DistillationBatch:
    item_count = len(steps_from_end)
    return DistillationBatch(
        obs={"image": torch.zeros(item_count, 1, 1, 1)},
        actions=torch.zeros(item_count, dtype=torch.long),
        teacher_logits=torch.zeros(item_count, 2),
        weights=torch.ones(item_count),
        accepted_episodes=1,
        episodes_seen=1,
        steps=item_count,
        mean_return=1.0,
        mean_length=float(item_count),
        disagreement=torch.as_tensor(disagreement, dtype=torch.float32) if disagreement is not None else None,
        steps_from_end=torch.as_tensor(steps_from_end, dtype=torch.long),
    )


def _spec(*, warmup_rounds: int) -> LearnerStateSpec:
    return LearnerStateSpec(
        name="test",
        output_dir="outputs/tests/lss_temporal_credit",
        teacher=TeacherConfig(config="teacher.yaml", checkpoint="teacher.pt"),
        student=StudentConfig(config="student.yaml", checkpoint="student.pt"),
        loop=LoopConfig(rounds=4, warmup_rounds=warmup_rounds),
        evaluation=EvalConfig(),
    )


def test_apply_weight_adjustments_last_step_only_zeroes_non_terminal_steps() -> None:
    student = StudentConfig(
        config="student.yaml",
        checkpoint="student.pt",
        temporal_credit_mode="last_step",
    )
    weights = _apply_weight_adjustments(student, _batch(steps_from_end=[3, 2, 1, 0]), torch.ones(4), torch.device("cpu"))
    assert weights.tolist() == [0.0, 0.0, 0.0, 1.0]


def test_apply_weight_adjustments_last_two_steps_keeps_penultimate_and_last() -> None:
    student = StudentConfig(
        config="student.yaml",
        checkpoint="student.pt",
        temporal_credit_mode="last_two_steps",
    )
    weights = _apply_weight_adjustments(student, _batch(steps_from_end=[3, 2, 1, 0]), torch.ones(4), torch.device("cpu"))
    assert weights.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_apply_weight_adjustments_stochastic_last_two_extremes_match_deterministic_modes() -> None:
    batch = _batch(steps_from_end=[2, 1, 0])
    last_only = StudentConfig(
        config="student.yaml",
        checkpoint="student.pt",
        temporal_credit_mode="stochastic_last_two",
        temporal_penultimate_keep_prob=0.0,
    )
    keep_two = StudentConfig(
        config="student.yaml",
        checkpoint="student.pt",
        temporal_credit_mode="stochastic_last_two",
        temporal_penultimate_keep_prob=1.0,
    )
    assert _apply_weight_adjustments(last_only, batch, torch.ones(3), torch.device("cpu")).tolist() == [0.0, 0.0, 1.0]
    assert _apply_weight_adjustments(keep_two, batch, torch.ones(3), torch.device("cpu")).tolist() == [0.0, 1.0, 1.0]


def test_warmup_only_round_uses_configured_prefix() -> None:
    spec = _spec(warmup_rounds=2)
    assert _warmup_only_round(spec, 1) is True
    assert _warmup_only_round(spec, 2) is True
    assert _warmup_only_round(spec, 3) is False


def test_dataset_weights_support_teacher_confidence_floor_and_ceiling() -> None:
    batch = DistillationBatch(
        obs={"image": torch.zeros(3, 1, 1, 1)},
        actions=torch.zeros(3, dtype=torch.long),
        teacher_logits=torch.zeros(3, 2),
        weights=torch.ones(3),
        accepted_episodes=1,
        episodes_seen=1,
        steps=3,
        mean_return=1.0,
        mean_length=3.0,
        teacher_confidence=torch.as_tensor([0.4, 0.8, 0.99], dtype=torch.float32),
    )
    student = StudentConfig(
        config="student.yaml",
        checkpoint="student.pt",
        weighting="teacher_confidence",
        teacher_confidence_floor=0.6,
        teacher_confidence_ceiling=0.9,
    )
    weights = _dataset_weights(student, batch, torch.device("cpu"))
    assert torch.allclose(weights, torch.as_tensor([0.6, 0.8, 0.9], dtype=torch.float32))
