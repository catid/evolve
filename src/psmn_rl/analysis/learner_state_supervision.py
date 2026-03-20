from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from psmn_rl.analysis.policy_distillation import (
    DistillationBatch,
    EvalConfig,
    StudentConfig,
    TeacherConfig,
    _best_sampled_mode,
    _capture_raw_obs,
    _dataset_weights,
    _discover_run_dirs,
    _format_float,
    _load_model,
    _set_trainable_parameters,
    _stack_obs,
    _student_loss,
    evaluate_modes,
)
from psmn_rl.config import dump_config, load_config
from psmn_rl.envs.registry import make_eval_env, make_reset_seeds
from psmn_rl.metrics import MetricAggregator, scalarize_metrics
from psmn_rl.rl.distributed.ddp import detect_device
from psmn_rl.rl.ppo.algorithm import _episode_successes, prepare_done, prepare_obs
from psmn_rl.utils.io import get_git_commit, get_git_dirty
from psmn_rl.utils.seed import set_seed


@dataclass(slots=True)
class LoopConfig:
    rounds: int = 4
    warmup_rounds: int = 0
    episodes_per_round: int = 64
    max_episodes_per_round: int = 96
    aggregation: str = "append_all"
    max_dataset_steps: int | None = None
    phase_quota_weights: dict[str, float] | None = None


@dataclass(slots=True)
class RoundCollection:
    batch: DistillationBatch
    diagnostics: dict[str, Any]


@dataclass(slots=True)
class LearnerStateSpec:
    name: str
    output_dir: str
    teacher: TeacherConfig
    student: StudentConfig
    loop: LoopConfig
    evaluation: EvalConfig


PHASE_NAMES: tuple[str, ...] = (
    "search_key",
    "at_key",
    "carry_key",
    "at_locked_door",
    "post_unlock",
)
PHASE_TO_ID: dict[str, int] = {name: index for index, name in enumerate(PHASE_NAMES)}
PHASE_ID_TO_NAME: dict[int, str] = {index: name for name, index in PHASE_TO_ID.items()}


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _load_spec(path: str | Path) -> LearnerStateSpec:
    raw = _load_yaml(path)
    return LearnerStateSpec(
        name=str(raw["name"]),
        output_dir=str(raw["output_dir"]),
        teacher=TeacherConfig(**raw["teacher"]),
        student=StudentConfig(**raw["student"]),
        loop=LoopConfig(**raw["loop"]),
        evaluation=EvalConfig(**raw.get("evaluation", {})),
    )


def _combine_optional_tensor(left: torch.Tensor | None, right: torch.Tensor | None) -> torch.Tensor | None:
    if left is None:
        return right
    if right is None:
        return left
    return torch.cat([left, right], dim=0)


def _concat_batches(left: DistillationBatch | None, right: DistillationBatch) -> DistillationBatch:
    if left is None:
        return right
    obs = {
        key: torch.cat([left.obs[key], right.obs[key]], dim=0)
        for key in left.obs
    }
    return DistillationBatch(
        obs=obs,
        actions=torch.cat([left.actions, right.actions], dim=0),
        teacher_logits=torch.cat([left.teacher_logits, right.teacher_logits], dim=0),
        weights=torch.cat([left.weights, right.weights], dim=0),
        teacher_confidence=_combine_optional_tensor(left.teacher_confidence, right.teacher_confidence),
        phase_ids=_combine_optional_tensor(left.phase_ids, right.phase_ids),
        disagreement=_combine_optional_tensor(left.disagreement, right.disagreement),
        steps_from_end=_combine_optional_tensor(left.steps_from_end, right.steps_from_end),
        accepted_episodes=left.accepted_episodes + right.accepted_episodes,
        episodes_seen=left.episodes_seen + right.episodes_seen,
        steps=int(left.steps + right.steps),
        mean_return=float(np.mean([left.mean_return, right.mean_return])),
        mean_length=float(np.mean([left.mean_length, right.mean_length])),
    )


def _slice_batch(batch: DistillationBatch, indices: torch.Tensor) -> DistillationBatch:
    return DistillationBatch(
        obs={key: value[indices] for key, value in batch.obs.items()},
        actions=batch.actions[indices],
        teacher_logits=batch.teacher_logits[indices],
        weights=batch.weights[indices],
        accepted_episodes=batch.accepted_episodes,
        episodes_seen=batch.episodes_seen,
        steps=int(indices.numel()),
        mean_return=batch.mean_return,
        mean_length=batch.mean_length,
        teacher_confidence=batch.teacher_confidence[indices] if batch.teacher_confidence is not None else None,
        phase_ids=batch.phase_ids[indices] if batch.phase_ids is not None else None,
        disagreement=batch.disagreement[indices] if batch.disagreement is not None else None,
        steps_from_end=batch.steps_from_end[indices] if batch.steps_from_end is not None else None,
    )


def _cap_recent_indices(total_steps: int, max_steps: int) -> torch.Tensor:
    start = max(total_steps - max_steps, 0)
    return torch.arange(start, total_steps, dtype=torch.long)


def _cap_recent_balanced_indices(actions: torch.Tensor, max_steps: int) -> torch.Tensor:
    action_count = int(actions.max().item()) + 1 if actions.numel() else 0
    if action_count <= 0:
        return torch.empty(0, dtype=torch.long)
    quota = max(max_steps // action_count, 1)
    kept: list[int] = []
    per_action: dict[int, int] = {action: 0 for action in range(action_count)}
    for index in range(actions.numel() - 1, -1, -1):
        action = int(actions[index].item())
        if per_action[action] >= quota:
            continue
        kept.append(index)
        per_action[action] += 1
        if len(kept) >= max_steps:
            break
    if len(kept) < max_steps:
        seen = set(kept)
        for index in range(actions.numel() - 1, -1, -1):
            if index in seen:
                continue
            kept.append(index)
            if len(kept) >= max_steps:
                break
    kept.sort()
    return torch.as_tensor(kept, dtype=torch.long)


def _phase_weight_lookup(weights: dict[str, float] | None, phase_name: str) -> float:
    if not weights:
        return 1.0
    return float(weights.get(phase_name, weights.get("default", 1.0)))


def _phase_balanced_recent_indices(
    phase_ids: torch.Tensor | None,
    max_steps: int,
    phase_quota_weights: dict[str, float] | None,
) -> torch.Tensor:
    if phase_ids is None or phase_ids.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    phases_present = sorted(
        {
            int(phase_id.item())
            for phase_id in phase_ids.unique(sorted=True)
            if int(phase_id.item()) in PHASE_ID_TO_NAME
        }
    )
    if not phases_present:
        return torch.empty(0, dtype=torch.long)
    raw_weights = {
        phase_id: max(_phase_weight_lookup(phase_quota_weights, PHASE_ID_TO_NAME[phase_id]), 0.0)
        for phase_id in phases_present
    }
    if sum(raw_weights.values()) <= 0.0:
        raw_weights = {phase_id: 1.0 for phase_id in phases_present}
    quotas = {
        phase_id: max(int(round(max_steps * (weight / sum(raw_weights.values())))), 1)
        for phase_id, weight in raw_weights.items()
    }
    kept: list[int] = []
    per_phase = {phase_id: 0 for phase_id in phases_present}
    for index in range(phase_ids.numel() - 1, -1, -1):
        phase_id = int(phase_ids[index].item())
        if phase_id not in per_phase:
            continue
        if per_phase[phase_id] >= quotas[phase_id]:
            continue
        kept.append(index)
        per_phase[phase_id] += 1
        if len(kept) >= max_steps:
            break
    if len(kept) < max_steps:
        seen = set(kept)
        for index in range(phase_ids.numel() - 1, -1, -1):
            if index in seen:
                continue
            kept.append(index)
            if len(kept) >= max_steps:
                break
    kept.sort()
    return torch.as_tensor(kept, dtype=torch.long)


def _apply_aggregation_policy(spec: LearnerStateSpec, batch: DistillationBatch) -> DistillationBatch:
    if spec.loop.aggregation == "append_all" or spec.loop.max_dataset_steps is None:
        batch.steps = int(batch.actions.numel())
        return batch
    max_steps = int(spec.loop.max_dataset_steps)
    if batch.actions.numel() <= max_steps:
        batch.steps = int(batch.actions.numel())
        return batch
    if spec.loop.aggregation == "cap_recent":
        keep = _cap_recent_indices(int(batch.actions.numel()), max_steps)
    elif spec.loop.aggregation == "cap_recent_balanced":
        keep = _cap_recent_balanced_indices(batch.actions, max_steps)
    elif spec.loop.aggregation == "phase_balanced_recent":
        keep = _phase_balanced_recent_indices(batch.phase_ids, max_steps, spec.loop.phase_quota_weights)
    else:
        raise ValueError(f"unsupported aggregation mode: {spec.loop.aggregation}")
    return _slice_batch(batch, keep)


def _extract_phase(env: Any) -> str:
    unwrapped = env.unwrapped
    carrying = unwrapped.carrying
    agent_pos = tuple(int(value) for value in unwrapped.agent_pos)
    door_pos = None
    key_pos = None
    door_locked = False
    door_open = False
    for x in range(int(unwrapped.width)):
        for y in range(int(unwrapped.height)):
            obj = unwrapped.grid.get(x, y)
            if obj is None:
                continue
            name = type(obj).__name__.lower()
            if name == "door":
                door_pos = (x, y)
                door_locked = bool(getattr(obj, "is_locked", False))
                door_open = bool(getattr(obj, "is_open", False))
            elif name == "key":
                key_pos = (x, y)

    def distance(position: tuple[int, int] | None) -> int | None:
        if position is None:
            return None
        return abs(agent_pos[0] - position[0]) + abs(agent_pos[1] - position[1])

    carrying_key = carrying is not None and type(carrying).__name__.lower() == "key"
    key_distance = distance(key_pos)
    door_distance = distance(door_pos)
    near_key = key_distance is not None and key_distance <= 1
    near_door = door_distance is not None and door_distance <= 1
    if carrying_key and door_locked:
        return "at_locked_door" if near_door else "carry_key"
    if carrying_key and (door_open or not door_locked):
        return "post_unlock"
    if not carrying_key and key_pos is not None:
        return "at_key" if near_key else "search_key"
    if door_open or not door_locked:
        return "post_unlock"
    return "search_key"


def _apply_weight_adjustments(spec: StudentConfig, dataset: DistillationBatch, weights: torch.Tensor, device: torch.device) -> torch.Tensor:
    adjusted = weights.clone()
    if spec.phase_weights:
        if dataset.phase_ids is None:
            raise ValueError("phase_weights requested but dataset lacks phase_ids")
        phase_ids = dataset.phase_ids.to(device)
        phase_multiplier = torch.full_like(adjusted, float(spec.default_phase_weight))
        for phase_name, phase_weight in spec.phase_weights.items():
            phase_id = PHASE_TO_ID.get(phase_name)
            if phase_id is None:
                raise ValueError(f"unsupported phase weight key: {phase_name}")
            phase_multiplier = torch.where(
                phase_ids == phase_id,
                torch.full_like(phase_multiplier, float(phase_weight)),
                phase_multiplier,
            )
        adjusted = adjusted * phase_multiplier
    if spec.disagreement_bonus:
        if dataset.disagreement is None:
            raise ValueError("disagreement_bonus requested but dataset lacks disagreement")
        adjusted = adjusted * (1.0 + dataset.disagreement.to(device) * float(spec.disagreement_bonus))
    temporal_credit_mode = str(spec.temporal_credit_mode)
    if temporal_credit_mode != "uniform":
        if dataset.steps_from_end is None:
            raise ValueError("temporal_credit_mode requested but dataset lacks steps_from_end")
        steps_from_end = dataset.steps_from_end.to(device)
        if temporal_credit_mode == "last_step":
            temporal_multiplier = (steps_from_end == 0).to(adjusted.dtype)
        elif temporal_credit_mode in {"last_two_steps", "final_plus_penultimate"}:
            temporal_multiplier = (steps_from_end <= 1).to(adjusted.dtype)
        elif temporal_credit_mode == "stochastic_last_two":
            keep_prob = float(spec.temporal_penultimate_keep_prob)
            if keep_prob < 0.0 or keep_prob > 1.0:
                raise ValueError("temporal_penultimate_keep_prob must be in [0, 1]")
            temporal_multiplier = (steps_from_end == 0).to(adjusted.dtype)
            if keep_prob > 0.0:
                penultimate_mask = steps_from_end == 1
                if keep_prob >= 1.0:
                    temporal_multiplier = temporal_multiplier + penultimate_mask.to(adjusted.dtype)
                else:
                    sampled = (torch.rand_like(adjusted) < keep_prob) & penultimate_mask
                    temporal_multiplier = temporal_multiplier + sampled.to(adjusted.dtype)
        else:
            raise ValueError(f"unsupported temporal_credit_mode: {temporal_credit_mode}")
        adjusted = adjusted * temporal_multiplier
    return adjusted


def _warmup_only_round(spec: LearnerStateSpec, round_index: int) -> bool:
    return round_index <= max(int(spec.loop.warmup_rounds), 0)


def _hash_raw_obs(obs_sample: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha1()
    for key in sorted(obs_sample):
        value = np.asarray(obs_sample[key])
        digest.update(key.encode("utf-8"))
        digest.update(value.shape.__repr__().encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _teacher_student_step_metrics(
    teacher_logits: torch.Tensor,
    teacher_action: torch.Tensor,
    student_logits: torch.Tensor,
    student_action: torch.Tensor,
) -> dict[str, np.ndarray]:
    teacher_probs = torch.softmax(teacher_logits, dim=-1)
    student_probs = torch.softmax(student_logits, dim=-1)
    teacher_top2 = torch.topk(teacher_logits, k=min(2, teacher_logits.size(-1)), dim=-1).values
    return {
        "teacher_confidence": teacher_probs.max(dim=-1).values.detach().cpu().numpy(),
        "teacher_entropy": (-(teacher_probs.clamp_min(1e-8) * teacher_probs.clamp_min(1e-8).log()).sum(dim=-1)).detach().cpu().numpy(),
        "teacher_margin": (teacher_top2[..., 0] - teacher_top2[..., 1] if teacher_top2.size(-1) > 1 else teacher_top2[..., 0]).detach().cpu().numpy(),
        "student_confidence": student_probs.max(dim=-1).values.detach().cpu().numpy(),
        "student_entropy": (-(student_probs.clamp_min(1e-8) * student_probs.clamp_min(1e-8).log()).sum(dim=-1)).detach().cpu().numpy(),
        "teacher_student_disagreement": (teacher_action != student_action).float().detach().cpu().numpy(),
    }


def _collect_teacher_labels_for_student_states(
    spec: LearnerStateSpec,
    teacher: Any,
    student: Any,
    device: torch.device,
    round_index: int,
) -> RoundCollection:
    env_config = load_config(spec.student.config)
    set_seed(env_config.seed + 30_000 + round_index, deterministic=env_config.system.deterministic)
    eval_env = make_eval_env(env_config.env, seed=env_config.seed + 300 + round_index, world_rank=0)
    reset_seed = make_reset_seeds(env_config.env.num_eval_envs, env_config.seed + 300 + round_index, world_rank=0)
    obs, _ = eval_env.reset(seed=reset_seed)
    obs_t = prepare_obs(obs, device)
    teacher_done = torch.ones(env_config.env.num_eval_envs, device=device, dtype=torch.bool)
    student_done = torch.ones(env_config.env.num_eval_envs, device=device, dtype=torch.bool)
    teacher_state = teacher.initial_state(env_config.env.num_eval_envs, device)
    student_state = student.initial_state(env_config.env.num_eval_envs, device)

    episode_obs: list[list[dict[str, np.ndarray]]] = [[] for _ in range(env_config.env.num_eval_envs)]
    episode_actions: list[list[int]] = [[] for _ in range(env_config.env.num_eval_envs)]
    episode_logits: list[list[np.ndarray]] = [[] for _ in range(env_config.env.num_eval_envs)]
    episode_teacher_confidence: list[list[float]] = [[] for _ in range(env_config.env.num_eval_envs)]
    episode_teacher_entropy: list[list[float]] = [[] for _ in range(env_config.env.num_eval_envs)]
    episode_teacher_margin: list[list[float]] = [[] for _ in range(env_config.env.num_eval_envs)]
    episode_student_confidence: list[list[float]] = [[] for _ in range(env_config.env.num_eval_envs)]
    episode_student_entropy: list[list[float]] = [[] for _ in range(env_config.env.num_eval_envs)]
    episode_disagreement: list[list[float]] = [[] for _ in range(env_config.env.num_eval_envs)]
    episode_phase_ids: list[list[int]] = [[] for _ in range(env_config.env.num_eval_envs)]
    episode_returns = np.zeros(env_config.env.num_eval_envs, dtype=np.float32)
    episode_lengths = np.zeros(env_config.env.num_eval_envs, dtype=np.int32)
    collected_obs: list[dict[str, np.ndarray]] = []
    collected_actions: list[int] = []
    collected_logits: list[np.ndarray] = []
    collected_weights: list[float] = []
    collected_teacher_confidence: list[float] = []
    collected_teacher_entropy: list[float] = []
    collected_teacher_margin: list[float] = []
    collected_student_confidence: list[float] = []
    collected_student_entropy: list[float] = []
    collected_disagreements: list[float] = []
    collected_phase_ids: list[int] = []
    collected_returns: list[float] = []
    collected_lengths: list[float] = []
    collected_success_flags: list[float] = []
    success_episode_disagreement: list[float] = []
    failure_episode_disagreement: list[float] = []
    collected_steps_from_end: list[int] = []
    accepted_episodes = 0
    episodes_seen = 0
    route_metrics = MetricAggregator()

    try:
        while accepted_episodes < spec.loop.episodes_per_round and episodes_seen < spec.loop.max_episodes_per_round:
            with torch.inference_mode():
                teacher_output = teacher(obs_t, state=teacher_state, done=teacher_done)
                if spec.teacher.greedy:
                    teacher_action = teacher_output.logits.argmax(dim=-1)
                else:
                    teacher_dist = teacher.get_dist(teacher_output.logits, temperature=spec.teacher.temperature)
                    teacher_action = teacher_dist.sample()
                teacher_state = teacher_output.next_state

                student_output = student(obs_t, state=student_state, done=student_done)
                student_action = student_output.logits.argmax(dim=-1)
                student_state = student_output.next_state

            step_diag = _teacher_student_step_metrics(
                teacher_output.logits,
                teacher_action,
                student_output.logits,
                student_action,
            )
            route_metrics.update(student_output.metrics)
            teacher_action_np = teacher_action.detach().cpu().numpy()
            teacher_logits_np = teacher_output.logits.detach().cpu().numpy()
            student_action_np = student_action.detach().cpu().numpy()
            raw_obs = _capture_raw_obs(obs)
            phase_names = [_extract_phase(eval_env.envs[index]) for index in range(env_config.env.num_eval_envs)]
            for index in range(env_config.env.num_eval_envs):
                episode_obs[index].append({key: value[index].copy() for key, value in raw_obs.items()})
                episode_actions[index].append(int(teacher_action_np[index]))
                episode_logits[index].append(teacher_logits_np[index].copy())
                episode_teacher_confidence[index].append(float(step_diag["teacher_confidence"][index]))
                episode_teacher_entropy[index].append(float(step_diag["teacher_entropy"][index]))
                episode_teacher_margin[index].append(float(step_diag["teacher_margin"][index]))
                episode_student_confidence[index].append(float(step_diag["student_confidence"][index]))
                episode_student_entropy[index].append(float(step_diag["student_entropy"][index]))
                episode_disagreement[index].append(float(step_diag["teacher_student_disagreement"][index]))
                episode_phase_ids[index].append(PHASE_TO_ID[phase_names[index]])

            next_obs, reward, terminated, truncated, info = eval_env.step(student_action_np)
            done_np = np.logical_or(terminated, truncated)
            step_success = _episode_successes(reward, done_np, info)
            episode_returns += reward
            episode_lengths += 1

            for index, finished in enumerate(done_np):
                if not finished:
                    continue
                episodes_seen += 1
                accepted_episodes += 1
                episode_return = float(episode_returns[index])
                collected_returns.append(episode_return)
                collected_lengths.append(float(episode_lengths[index]))
                collected_success_flags.append(float(step_success[index]))
                collected_obs.extend(episode_obs[index])
                collected_actions.extend(episode_actions[index])
                collected_logits.extend(episode_logits[index])
                collected_weights.extend([episode_return] * len(episode_actions[index]))
                collected_teacher_confidence.extend(episode_teacher_confidence[index])
                collected_teacher_entropy.extend(episode_teacher_entropy[index])
                collected_teacher_margin.extend(episode_teacher_margin[index])
                collected_student_confidence.extend(episode_student_confidence[index])
                collected_student_entropy.extend(episode_student_entropy[index])
                collected_disagreements.extend(episode_disagreement[index])
                collected_phase_ids.extend(episode_phase_ids[index])
                collected_steps_from_end.extend(reversed(range(len(episode_actions[index]))))
                mean_disagreement = float(np.mean(episode_disagreement[index])) if episode_disagreement[index] else 0.0
                if step_success[index] > 0.0:
                    success_episode_disagreement.append(mean_disagreement)
                else:
                    failure_episode_disagreement.append(mean_disagreement)
                episode_obs[index].clear()
                episode_actions[index].clear()
                episode_logits[index].clear()
                episode_teacher_confidence[index].clear()
                episode_teacher_entropy[index].clear()
                episode_teacher_margin[index].clear()
                episode_student_confidence[index].clear()
                episode_student_entropy[index].clear()
                episode_disagreement[index].clear()
                episode_phase_ids[index].clear()
                episode_returns[index] = 0.0
                episode_lengths[index] = 0

            obs = next_obs
            obs_t = prepare_obs(obs, device)
            teacher_done = prepare_done(done_np, device)
            student_done = prepare_done(done_np, device)
    finally:
        eval_env.close()

    if not collected_obs:
        raise RuntimeError("learner-state supervision collected no labeled states")

    state_hashes = {_hash_raw_obs(sample) for sample in collected_obs}
    action_count = int(max(collected_actions) + 1) if collected_actions else 0
    action_hist = np.bincount(np.asarray(collected_actions, dtype=np.int64), minlength=action_count) if action_count > 0 else np.zeros(0, dtype=np.int64)
    action_probs = action_hist.astype(np.float64)
    if action_probs.sum() > 0:
        action_probs = action_probs / action_probs.sum()
        action_entropy = float(-(action_probs[action_probs > 0.0] * np.log(action_probs[action_probs > 0.0])).sum())
    else:
        action_entropy = 0.0

    teacher_logits_t = torch.as_tensor(np.stack(collected_logits, axis=0), dtype=torch.float32)
    batch = DistillationBatch(
        obs=_stack_obs(collected_obs),
        actions=torch.as_tensor(collected_actions, dtype=torch.long),
        teacher_logits=teacher_logits_t,
        weights=torch.as_tensor(collected_weights, dtype=torch.float32),
        teacher_confidence=torch.as_tensor(collected_teacher_confidence, dtype=torch.float32),
        phase_ids=torch.as_tensor(collected_phase_ids, dtype=torch.long),
        disagreement=torch.as_tensor(collected_disagreements, dtype=torch.float32),
        steps_from_end=torch.as_tensor(collected_steps_from_end, dtype=torch.long),
        accepted_episodes=accepted_episodes,
        episodes_seen=episodes_seen,
        steps=len(collected_actions),
        mean_return=float(np.mean(collected_returns)) if collected_returns else 0.0,
        mean_length=float(np.mean(collected_lengths)) if collected_lengths else 0.0,
    )
    diagnostics: dict[str, Any] = {
        "collection/episodes_seen": float(episodes_seen),
        "collection/accepted_episodes": float(accepted_episodes),
        "collection/steps": float(batch.steps),
        "collection/mean_return": float(np.mean(collected_returns)) if collected_returns else 0.0,
        "collection/mean_length": float(np.mean(collected_lengths)) if collected_lengths else 0.0,
        "collection/success_rate": float(np.mean(collected_success_flags)) if collected_success_flags else 0.0,
        "collection/unique_state_count": float(len(state_hashes)),
        "collection/unique_state_ratio": float(len(state_hashes) / max(len(collected_obs), 1)),
        "collection/teacher_confidence_mean": float(np.mean(collected_teacher_confidence)) if collected_teacher_confidence else 0.0,
        "collection/teacher_confidence_std": float(np.std(collected_teacher_confidence)) if collected_teacher_confidence else 0.0,
        "collection/teacher_entropy_mean": float(np.mean(collected_teacher_entropy)) if collected_teacher_entropy else 0.0,
        "collection/teacher_margin_mean": float(np.mean(collected_teacher_margin)) if collected_teacher_margin else 0.0,
        "collection/student_confidence_mean": float(np.mean(collected_student_confidence)) if collected_student_confidence else 0.0,
        "collection/student_entropy_mean": float(np.mean(collected_student_entropy)) if collected_student_entropy else 0.0,
        "collection/disagreement_rate": float(np.mean(collected_disagreements)) if collected_disagreements else 0.0,
        "collection/success_episode_disagreement": float(np.mean(success_episode_disagreement)) if success_episode_disagreement else None,
        "collection/failure_episode_disagreement": float(np.mean(failure_episode_disagreement)) if failure_episode_disagreement else None,
        "collection/steps_from_end_mean": float(np.mean(collected_steps_from_end)) if collected_steps_from_end else 0.0,
        "collection/last_step_frac": float(np.mean(np.asarray(collected_steps_from_end, dtype=np.int64) == 0)) if collected_steps_from_end else 0.0,
        "collection/last_two_step_frac": float(np.mean(np.asarray(collected_steps_from_end, dtype=np.int64) <= 1)) if collected_steps_from_end else 0.0,
        "collection/action_entropy": action_entropy,
    }
    for action_index, count in enumerate(action_hist):
        diagnostics[f"collection/action_count_{action_index}"] = float(count)
        diagnostics[f"collection/action_frac_{action_index}"] = float(count / max(len(collected_actions), 1))
    if collected_phase_ids:
        phase_hist = np.bincount(np.asarray(collected_phase_ids, dtype=np.int64), minlength=len(PHASE_NAMES))
        for phase_index, phase_name in enumerate(PHASE_NAMES):
            count = float(phase_hist[phase_index])
            diagnostics[f"collection/phase_count_{phase_name}"] = count
            diagnostics[f"collection/phase_frac_{phase_name}"] = float(count / max(len(collected_phase_ids), 1))
    diagnostics.update({f"collection/{key}": value for key, value in route_metrics.compute().items()})
    return RoundCollection(batch=batch, diagnostics=diagnostics)


def _fine_tune_student(
    spec: LearnerStateSpec,
    student: Any,
    dataset: DistillationBatch,
    device: torch.device,
) -> dict[str, float]:
    student.train()
    trainable_params = _set_trainable_parameters(student, spec.student.target)
    optimizer = torch.optim.Adam(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=spec.student.learning_rate,
    )
    obs = {key: value.to(device) for key, value in dataset.obs.items()}
    actions = dataset.actions.to(device)
    teacher_logits = dataset.teacher_logits.to(device)
    weights = _apply_weight_adjustments(spec.student, dataset, _dataset_weights(spec.student, dataset, device), device)
    done = torch.zeros(actions.size(0), device=device, dtype=torch.bool)
    losses: list[float] = []

    for _epoch in range(spec.student.epochs):
        indices = torch.randperm(actions.size(0), device=device)
        for start in range(0, actions.size(0), spec.student.batch_size):
            batch_index = indices[start : start + spec.student.batch_size]
            obs_batch = {key: value[batch_index] for key, value in obs.items()}
            action_batch = actions[batch_index]
            teacher_logits_batch = teacher_logits[batch_index]
            done_batch = done[batch_index]
            weight_batch = weights[batch_index]
            output = student(obs_batch, state={}, done=done_batch)
            loss = _student_loss(
                spec.student,
                output.logits,
                action_batch,
                teacher_logits_batch,
                weight_batch,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

    student.eval()
    return {
        "fine_tune/steps": float(len(losses)),
        "fine_tune/loss_final": float(losses[-1]) if losses else 0.0,
        "fine_tune/loss_mean": float(np.mean(losses)) if losses else 0.0,
        "fine_tune/trainable_params": float(trainable_params),
    }


def _write_single_run_summary(
    output_dir: Path,
    spec: LearnerStateSpec,
    teacher_variant: str,
    student_variant: str,
    before: dict[str, dict[str, float]],
    rounds: list[dict[str, Any]],
) -> None:
    final_metrics = rounds[-1]
    best_round = max(rounds, key=lambda row: float(row["after_greedy_success"]))
    before_best_mode, before_best_sampled = _best_sampled_mode(before)
    summary = {
        "name": spec.name,
        "teacher_variant": teacher_variant,
        "student_variant": student_variant,
        "teacher_config": spec.teacher.config,
        "teacher_checkpoint": spec.teacher.checkpoint,
        "student_config": spec.student.config,
        "student_checkpoint": spec.student.checkpoint,
        "target": spec.student.target,
        "loss": spec.student.loss,
        "weighting": spec.student.weighting,
        "phase_weights": spec.student.phase_weights,
        "default_phase_weight": spec.student.default_phase_weight,
        "disagreement_bonus": spec.student.disagreement_bonus,
        "temporal_credit_mode": spec.student.temporal_credit_mode,
        "temporal_penultimate_keep_prob": spec.student.temporal_penultimate_keep_prob,
        "aggregation": spec.loop.aggregation,
        "warmup_rounds": spec.loop.warmup_rounds,
        "max_dataset_steps": spec.loop.max_dataset_steps,
        "phase_quota_weights": spec.loop.phase_quota_weights,
        "before_greedy_success": float(before["greedy"].get("eval_success_rate", 0.0)),
        "before_best_sampled_mode": before_best_mode,
        "before_best_sampled_success": before_best_sampled,
        "final_greedy_success": float(final_metrics["after_greedy_success"]),
        "final_best_sampled_success": float(final_metrics["after_best_sampled_success"]),
        "best_round_index": int(best_round["round"]),
        "best_round_greedy_success": float(best_round["after_greedy_success"]),
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "rounds": rounds,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Learner-State Supervision Summary",
        "",
        f"- teacher_variant: `{teacher_variant}`",
        f"- student_variant: `{student_variant}`",
        f"- target: `{spec.student.target}`",
        f"- loss: `{spec.student.loss}`",
        f"- aggregation: `{spec.loop.aggregation}`",
        f"- weighting: `{spec.student.weighting}`",
        f"- temporal_credit_mode: `{spec.student.temporal_credit_mode}`",
        f"- warmup_rounds: `{int(spec.loop.warmup_rounds)}`",
        f"- before_greedy_success: `{summary['before_greedy_success']:.3f}`",
        f"- best_round_greedy_success: `{summary['best_round_greedy_success']:.3f}`",
        f"- final_greedy_success: `{summary['final_greedy_success']:.3f}`",
        "",
        "| Round | Added Episodes | Added Steps | Aggregate Steps | After Greedy | After Best Sampled | Disagreement | Teacher Conf | Unique Ratio | Loss Mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rounds:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["round"])),
                    str(int(row["added_episodes"])),
                    str(int(row["added_steps"])),
                    str(int(row["aggregate_steps"])),
                    f"{row['after_greedy_success']:.3f}",
                    f"{row['after_best_sampled_success']:.3f}",
                    f"{row['collection/disagreement_rate']:.3f}",
                    f"{row['collection/teacher_confidence_mean']:.3f}",
                    f"{row['collection/unique_state_ratio']:.3f}",
                    f"{row['fine_tune/loss_mean']:.3f}",
                ]
            )
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Learner-State Supervision Report",
        "",
        "## Summary",
        "",
        "| Run | Teacher | Student | Target | Loss | Aggregation | Before Greedy | Best Round Greedy | Final Greedy | Best Sampled After | Rounds |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (str(item["student_variant"]), str(item["teacher_variant"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['run_name']}`",
                    str(row["teacher_variant"]),
                    str(row["student_variant"]),
                    str(row["target"]),
                    str(row.get("loss", "ce")),
                    str(row.get("aggregation", "append_all")),
                    _format_float(row["before_greedy_success"]),
                    _format_float(row["best_round_greedy_success"]),
                    _format_float(row["final_greedy_success"]),
                    _format_float(row["final_best_sampled_success"]),
                    str(len(row["rounds"])),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", ""])
    for row in sorted(rows, key=lambda item: (str(item["student_variant"]), str(item["teacher_variant"]))):
        lines.append(
            f"- `{row['run_name']}` uses learner-state supervision from `{row['teacher_variant']}` into `{row['student_variant']}`, "
            f"moving greedy success from `{row['before_greedy_success']:.3f}` to best-round `{row['best_round_greedy_success']:.3f}` "
            f"and final `{row['final_greedy_success']:.3f}` with loss `{row.get('loss', 'ce')}` "
            f"and aggregation `{row.get('aggregation', 'append_all')}`."
        )
    return "\n".join(lines) + "\n"


def run_once(args: argparse.Namespace) -> None:
    spec_path = Path(args.spec)
    spec = _load_spec(spec_path)
    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "spec.yaml").write_text(spec_path.read_text(encoding="utf-8"), encoding="utf-8")

    teacher_config, teacher = _load_model(spec.teacher.config, spec.teacher.checkpoint, detect_device(args.device))
    student_config, student = _load_model(spec.student.config, spec.student.checkpoint, detect_device(args.device))
    dump_config(teacher_config, output_dir / "teacher_resolved_config.yaml")
    dump_config(student_config, output_dir / "student_resolved_config.yaml")
    dump_config(student_config, output_dir / "resolved_config.yaml")
    before = evaluate_modes(spec.student.config, student, args.device, spec.evaluation.episodes)

    aggregate: DistillationBatch | None = None
    round_rows: list[dict[str, Any]] = []
    for round_index in range(1, spec.loop.rounds + 1):
        round_collection = _collect_teacher_labels_for_student_states(
            spec=spec,
            teacher=teacher,
            student=student,
            device=detect_device(args.device),
            round_index=round_index,
        )
        round_batch = round_collection.batch
        aggregate = _apply_aggregation_policy(spec, _concat_batches(aggregate, round_batch))
        if _warmup_only_round(spec, round_index):
            fine_tune_metrics = {
                "fine_tune/steps": 0.0,
                "fine_tune/loss_final": 0.0,
                "fine_tune/loss_mean": 0.0,
                "fine_tune/trainable_params": 0.0,
                "fine_tune/warmup_only_round": 1.0,
            }
        else:
            fine_tune_metrics = _fine_tune_student(
                spec=spec,
                student=student,
                dataset=aggregate,
                device=detect_device(args.device),
            )
            fine_tune_metrics["fine_tune/warmup_only_round"] = 0.0
        after = evaluate_modes(spec.student.config, student, args.device, spec.evaluation.episodes)
        after_best_mode, after_best_sampled = _best_sampled_mode(after)
        round_row = {
            "round": float(round_index),
            "added_episodes": float(round_batch.accepted_episodes),
            "added_steps": float(round_batch.steps),
            "aggregate_steps": float(aggregate.steps),
            "after_greedy_success": float(after["greedy"].get("eval_success_rate", 0.0)),
            "after_greedy_return": float(after["greedy"].get("eval_return", 0.0)),
            "after_best_sampled_mode": after_best_mode,
            "after_best_sampled_success": after_best_sampled,
            "loss": spec.student.loss,
            "aggregation": spec.loop.aggregation,
            "target": spec.student.target,
            "weighting": spec.student.weighting,
            **fine_tune_metrics,
            **round_collection.diagnostics,
        }
        round_rows.append(round_row)
        torch.save(
            {
                "obs": aggregate.obs,
                "actions": aggregate.actions,
                "teacher_logits": aggregate.teacher_logits,
                "weights": aggregate.weights,
                "teacher_confidence": aggregate.teacher_confidence,
                "phase_ids": aggregate.phase_ids,
                "disagreement": aggregate.disagreement,
                "steps_from_end": aggregate.steps_from_end,
                "accepted_episodes": aggregate.accepted_episodes,
                "episodes_seen": aggregate.episodes_seen,
                "steps": aggregate.steps,
                "mean_return": aggregate.mean_return,
                "mean_length": aggregate.mean_length,
                "round_diagnostics": round_collection.diagnostics,
            },
            output_dir / f"round_{round_index:02d}_dataset.pt",
        )
        checkpoint = torch.load(spec.student.checkpoint, map_location="cpu", weights_only=False)
        checkpoint["model"] = student.state_dict()
        checkpoint["learner_state_supervision"] = {
            "round": round_index,
            "target": spec.student.target,
            "loss": spec.student.loss,
            "aggregation": spec.loop.aggregation,
            "weighting": spec.student.weighting,
            "temporal_credit_mode": spec.student.temporal_credit_mode,
            "temporal_penultimate_keep_prob": spec.student.temporal_penultimate_keep_prob,
            "phase_weights": spec.student.phase_weights,
            "default_phase_weight": spec.student.default_phase_weight,
            "disagreement_bonus": spec.student.disagreement_bonus,
            "warmup_rounds": spec.loop.warmup_rounds,
            "phase_quota_weights": spec.loop.phase_quota_weights,
            **fine_tune_metrics,
            **round_collection.diagnostics,
        }
        torch.save(checkpoint, output_dir / f"round_{round_index:02d}.pt")

    teacher.eval()
    student.eval()
    torch.save(
        {
            **torch.load(spec.student.checkpoint, map_location="cpu", weights_only=False),
            "model": student.state_dict(),
        },
        output_dir / "latest.pt",
    )
    _write_single_run_summary(
        output_dir=output_dir,
        spec=spec,
        teacher_variant=teacher_config.model.variant,
        student_variant=student_config.model.variant,
        before=before,
        rounds=round_rows,
    )


def build_report(args: argparse.Namespace) -> None:
    run_dirs = _discover_run_dirs(args.paths)
    if not run_dirs:
        raise SystemExit("no learner-state supervision runs found")
    rows = []
    for run_dir in run_dirs:
        row = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        row["run_name"] = run_dir.name
        rows.append(row)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_build_report(rows), encoding="utf-8")
    if args.csv is not None:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row if key != "rounds"})
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: value for key, value in row.items() if key in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run learner-state teacher supervision experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--spec", required=True)
    run_parser.add_argument("--device", default="cpu")

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("paths", nargs="+")
    report_parser.add_argument("--output", required=True)
    report_parser.add_argument("--csv", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run_once(args)
        return
    build_report(args)


if __name__ == "__main__":
    main()
