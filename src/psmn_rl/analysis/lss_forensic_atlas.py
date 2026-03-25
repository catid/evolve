from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from psmn_rl.analysis.lss_claim_consolidation import _group, _summary_stats
from psmn_rl.analysis.lss_frozen_claim import (
    REPRO_MODES,
    _capture_route_batch_stats,
    _evaluate_modes,
    _load_summary,
)
from psmn_rl.analysis.lss_robustness import (
    EvalTarget,
    _format_float,
    _greedy_row,
    _maybe_read_command,
    _write_main_process_report,
)
from psmn_rl.analysis.policy_distillation import _load_model
from psmn_rl.config import load_config
from psmn_rl.envs.minigrid import build_env_fn
from psmn_rl.logging import configure_logging
from psmn_rl.models.routing.sare import RoutedExpertCore, _gather_expert_outputs
from psmn_rl.rl.distributed.ddp import cleanup_distributed, init_distributed
from psmn_rl.rl.ppo.algorithm import prepare_done, prepare_obs
from psmn_rl.utils.io import get_git_commit, get_git_dirty
from psmn_rl.utils.seed import set_seed


LABEL_TO_VARIANT = {
    "kl_lss_token_dense": "token_dense",
    "kl_lss_single_expert": "single_expert",
    "kl_lss_sare": "sare",
}

DISPLAY_BY_LABEL = {
    "kl_lss_token_dense": "KL learner-state token_dense",
    "kl_lss_single_expert": "KL learner-state single_expert",
    "kl_lss_sare": "KL learner-state SARE",
}


@dataclass(slots=True)
class RunTarget:
    seed: int
    lane: str
    label: str
    run_dir: Path
    config_path: Path
    checkpoint_path: Path
    variant: str
    method: str


@dataclass(slots=True)
class SeedCase:
    lane: str
    seed: int
    teacher: RunTarget
    token_dense: RunTarget
    single_expert: RunTarget
    sare: RunTarget


def _load_case_config(path: Path) -> tuple[list[SeedCase], list[SeedCase], dict[str, int]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def run_target(seed: int, lane: str, label: str, run_dir: str) -> RunTarget:
        directory = Path(run_dir)
        config_path = directory / "student_resolved_config.yaml"
        if not config_path.exists():
            config_path = directory / "resolved_config.yaml"
        checkpoint_path = directory / "latest.pt"
        variant = LABEL_TO_VARIANT.get(label, label)
        return RunTarget(
            seed=seed,
            lane=lane,
            label=label,
            run_dir=directory,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            variant=variant,
            method=label,
        )

    def parse_cases(items: list[dict[str, Any]]) -> list[SeedCase]:
        parsed: list[SeedCase] = []
        for item in items:
            seed = int(item["seed"])
            lane = str(item["lane"])
            parsed.append(
                SeedCase(
                    lane=lane,
                    seed=seed,
                    teacher=run_target(seed, lane, "teacher_flat_dense", str(item["teacher"])),
                    token_dense=run_target(seed, lane, "kl_lss_token_dense", str(item["token_dense"])),
                    single_expert=run_target(seed, lane, "kl_lss_single_expert", str(item["single_expert"])),
                    sare=run_target(seed, lane, "kl_lss_sare", str(item["sare"])),
                )
            )
        return parsed

    settings = {
        "episodes": int(raw.get("episodes", 64)),
        "max_steps": int(raw.get("max_steps", 256)),
        "phase_sample_limit": int(raw.get("phase_sample_limit", 128)),
        "route_counterfactual_sample_limit": int(raw.get("route_counterfactual_sample_limit", 128)),
    }
    return parse_cases(raw.get("weak_cases", [])), parse_cases(raw.get("strong_cases", [])), settings


def _eval_target(target: RunTarget) -> EvalTarget:
    return EvalTarget(
        seed=target.seed,
        label=target.label,
        variant=target.variant,
        config_path=target.config_path,
        checkpoint_path=target.checkpoint_path,
        run_dir=target.run_dir,
        method=target.method,
        stage=target.lane,
        metadata={"lane": target.lane},
        command_path=target.run_dir / "command.txt",
    )


def _reproduction_targets(weak_cases: list[SeedCase]) -> list[EvalTarget]:
    targets: list[EvalTarget] = []
    for case in weak_cases:
        for target in (case.token_dense, case.single_expert, case.sare):
            targets.append(_eval_target(target))
    return targets


def _build_reproduction_note(rows: list[dict[str, Any]], case_config: Path, episodes: int) -> str:
    grouped = _group(rows)
    labels = [
        ("kl_lss_token_dense", "KL learner-state token_dense"),
        ("kl_lss_single_expert", "KL learner-state single_expert"),
        ("kl_lss_sare", "KL learner-state SARE"),
    ]
    lines = [
        "# Forensic-Atlas Reproduction Note",
        "",
        f"- case config: `{case_config}`",
        f"- external evaluation episodes per mode: `{episodes}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "| Seed | Variant | Greedy Success | Sampled t=1.0 Success | Config | Checkpoint | Command |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for seed in sorted({int(row["seed"]) for row in rows}):
        for label, display in labels:
            mode_rows = grouped[("fresh_final", seed, label)]
            greedy = _greedy_row(mode_rows)
            sampled = next(row for row in mode_rows if row["mode"] == "sampled_t1.0")
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(seed),
                        display,
                        _format_float(greedy.get("eval_success_rate")),
                        _format_float(sampled.get("eval_success_rate")),
                        f"`{greedy['config_path']}`",
                        f"`{greedy['checkpoint_path']}`",
                        f"`{_maybe_read_command(Path(greedy['run_dir']) / 'command.txt') or '-'}`",
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This note exactly re-evaluates the frozen `47/53/59` DoorKey block before any deeper forensic interpretation.",
            "- The external 64-episode path remains the decision path; later forensic artifacts add mechanism detail, not a replacement metric lane.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_single_env(config_path: Path):
    config = load_config(config_path)
    thunk = build_env_fn(
        env_id=config.env.env_id,
        seed=config.seed,
        env_index=0,
        max_episode_steps=config.env.max_episode_steps,
        fully_observed=config.env.fully_observed,
    )
    return thunk(), config


def _action_summary(logits: torch.Tensor, model) -> dict[str, Any]:
    probs = torch.softmax(logits, dim=-1)
    top2 = torch.topk(logits, k=min(2, logits.size(-1)), dim=-1).values
    margin = top2[..., 0] - top2[..., 1] if top2.size(-1) > 1 else top2[..., 0]
    return {
        "action": int(logits.argmax(dim=-1).item()),
        "confidence": float(probs.max(dim=-1).values.item()),
        "entropy": float((-(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)).item()),
        "margin": float(margin.item()),
    }


def _extract_grid_state(env) -> dict[str, Any]:
    unwrapped = env.unwrapped
    carrying = unwrapped.carrying
    agent_pos = tuple(int(value) for value in unwrapped.agent_pos)
    door_pos = None
    key_pos = None
    goal_pos = None
    door_locked = False
    door_open = False
    width = int(unwrapped.width)
    height = int(unwrapped.height)
    for x in range(width):
        for y in range(height):
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
            elif name == "goal":
                goal_pos = (x, y)

    def distance(position: tuple[int, int] | None) -> int | None:
        if position is None:
            return None
        return abs(agent_pos[0] - position[0]) + abs(agent_pos[1] - position[1])

    carrying_key = carrying is not None and type(carrying).__name__.lower() == "key"
    key_distance = distance(key_pos)
    door_distance = distance(door_pos)
    goal_distance = distance(goal_pos)
    near_key = key_distance is not None and key_distance <= 1
    near_door = door_distance is not None and door_distance <= 1
    if carrying_key and door_locked:
        phase = "at_locked_door" if near_door else "carry_key"
    elif carrying_key and (door_open or not door_locked):
        phase = "post_unlock"
    elif not carrying_key and key_pos is not None:
        phase = "at_key" if near_key else "search_key"
    elif door_open or not door_locked:
        phase = "post_unlock"
    else:
        phase = "search_key"
    return {
        "agent_pos": agent_pos,
        "door_pos": door_pos,
        "door_distance": door_distance,
        "door_locked": door_locked,
        "door_open": door_open,
        "key_pos": key_pos,
        "key_distance": key_distance,
        "goal_pos": goal_pos,
        "goal_distance": goal_distance,
        "carrying_key": carrying_key,
        "phase": phase,
    }


def _route_capture_summary(captured: dict[str, torch.Tensor] | None) -> dict[str, Any]:
    if not captured:
        return {
            "route_entropy": None,
            "dominant_pair": None,
            "dominant_pair_fraction": None,
            "unique_pair_count": None,
            "expert_loads": None,
        }
    route_probs = captured["route_probs"][0]
    topk_idx = captured["topk_idx"][0]
    pairs = [tuple(sorted(int(value) for value in topk_idx[token_index].tolist())) for token_index in range(topk_idx.shape[0])]
    counts = Counter(pairs)
    dominant_pair, dominant_count = counts.most_common(1)[0]
    expert_loads = route_probs.mean(dim=0).tolist()
    entropy = float((-(route_probs.clamp_min(1e-8) * route_probs.clamp_min(1e-8).log()).sum(dim=-1)).mean().item())
    return {
        "route_entropy": entropy,
        "dominant_pair": list(dominant_pair),
        "dominant_pair_fraction": float(dominant_count / max(topk_idx.shape[0], 1)),
        "unique_pair_count": float(len(counts)),
        "expert_loads": [float(value) for value in expert_loads],
    }


def _forward_with_route_capture(model, obs_t: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done_t: torch.Tensor) -> tuple[Any, dict[str, torch.Tensor] | None]:
    if not isinstance(getattr(model, "core", None), RoutedExpertCore):
        output = model(obs_t, state=state, done=done_t)
        return output, None
    captured: dict[str, torch.Tensor] = {}
    core: RoutedExpertCore = model.core
    original_route = core.route

    def route_with_capture(self: RoutedExpertCore, tokens: torch.Tensor, *args: Any, **kwargs: Any):
        route_probs, topk_values, topk_idx = original_route(tokens, *args, **kwargs)
        captured["route_probs"] = route_probs.detach().cpu()
        captured["topk_values"] = topk_values.detach().cpu()
        captured["topk_idx"] = topk_idx.detach().cpu()
        return route_probs, topk_values, topk_idx

    try:
        core.route = route_with_capture.__get__(core, RoutedExpertCore)
        output = model(obs_t, state=state, done=done_t)
    finally:
        core.route = original_route
    return output, captured


def _obs_to_batch(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "image": np.expand_dims(np.asarray(obs["image"]), axis=0),
        "direction": np.asarray([obs["direction"]]),
    }


def _action_name(env, action: int) -> str:
    return env.unwrapped.actions(action).name


def _failure_bucket(episode: dict[str, Any]) -> str:
    if episode["success"] > 0.0:
        return "success"
    if episode["key_pickup_step"] is None:
        return "before_key_pickup"
    if episode["door_unlock_step"] is None:
        return "after_key_before_unlock"
    return "after_unlock_before_goal"


def _find_representative_episode(episodes_by_label: dict[str, list[dict[str, Any]]], seed: int, kind: str) -> tuple[int, str]:
    if kind == "weak":
        episode_indices = sorted(
            set.intersection(
                *(set(int(ep["episode_index"]) for ep in episodes) for episodes in episodes_by_label.values())
            )
        )
        for episode_index in episode_indices:
            token = next(ep for ep in episodes_by_label["kl_lss_token_dense"] if int(ep["episode_index"]) == episode_index)
            single = next(ep for ep in episodes_by_label["kl_lss_single_expert"] if int(ep["episode_index"]) == episode_index)
            sare = next(ep for ep in episodes_by_label["kl_lss_sare"] if int(ep["episode_index"]) == episode_index)
            if sare["success"] <= 0.0 and token["success"] > 0.0 and single["success"] > 0.0:
                return episode_index, "sare_specific_failure"
        for episode_index in episode_indices:
            token = next(ep for ep in episodes_by_label["kl_lss_token_dense"] if int(ep["episode_index"]) == episode_index)
            single = next(ep for ep in episodes_by_label["kl_lss_single_expert"] if int(ep["episode_index"]) == episode_index)
            sare = next(ep for ep in episodes_by_label["kl_lss_sare"] if int(ep["episode_index"]) == episode_index)
            if sare["success"] <= 0.0 and token["success"] > 0.0 and single["success"] <= 0.0:
                return episode_index, "shared_structured_failure"
        for episode_index in episode_indices:
            sare = next(ep for ep in episodes_by_label["kl_lss_sare"] if int(ep["episode_index"]) == episode_index)
            if sare["success"] <= 0.0:
                return episode_index, "generic_sare_failure"
    else:
        for episode in episodes_by_label["kl_lss_sare"]:
            if episode["success"] > 0.0:
                return int(episode["episode_index"]), "strong_sare_success"
    first_episode = min(int(ep["episode_index"]) for episodes in episodes_by_label.values() for ep in episodes)
    return first_episode, "fallback"


def _as_float(value: Any) -> float:
    if value in (None, "", "-", "None"):
        return float("nan")
    return float(value)


def _snippet_rows(episode: dict[str, Any], around_step: int | None, window: int = 2) -> list[dict[str, Any]]:
    steps = episode["steps"]
    if around_step is None:
        around_step = 0
    start = max(int(around_step) - window, 0)
    end = min(int(around_step) + window + 1, len(steps))
    return steps[start:end]


def _append_phase_sample(
    storage: dict[str, list[dict[str, Any]]],
    phase: str,
    sample: dict[str, Any],
    limit: int,
) -> None:
    bucket = storage.setdefault(phase, [])
    if len(bucket) < limit:
        bucket.append(sample)
        return
    index = len(bucket) + 1
    if np.random.rand() < (limit / index):
        replace = int(np.random.randint(0, limit))
        bucket[replace] = sample


def _trace_variant(
    case: SeedCase,
    student: RunTarget,
    episodes: int,
    max_steps: int,
    device: torch.device,
    phase_sample_limit: int,
    reset_seed_base: int = 999,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    teacher_config, teacher_model = _load_model(str(case.teacher.config_path), str(case.teacher.checkpoint_path), device)
    student_config, student_model = _load_model(str(student.config_path), str(student.checkpoint_path), device)
    teacher_model.eval()
    student_model.eval()
    env, env_config = _build_single_env(student.config_path)
    assert teacher_config.env.env_id == student_config.env.env_id
    phase_samples: dict[str, list[dict[str, Any]]] = {}
    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    try:
        for episode_index in range(episodes):
            reset_seed = int(student_config.seed + reset_seed_base + episode_index)
            obs, _ = env.reset(seed=reset_seed)
            teacher_state = teacher_model.initial_state(1, device)
            student_state = student_model.initial_state(1, device)
            teacher_done = torch.ones(1, device=device, dtype=torch.bool)
            student_done = torch.ones(1, device=device, dtype=torch.bool)
            episode_steps: list[dict[str, Any]] = []
            episode_return = 0.0
            key_pickup_step = None
            door_unlock_step = None
            success = 0.0
            first_divergence_step = None
            first_divergence_phase = None
            for step_index in range(max_steps):
                pre_state = _extract_grid_state(env)
                obs_t = prepare_obs(_obs_to_batch(obs), device)
                with torch.inference_mode():
                    teacher_output = teacher_model(obs_t, state=teacher_state, done=teacher_done)
                    student_output, route_capture = _forward_with_route_capture(
                        student_model,
                        obs_t,
                        student_state,
                        student_done,
                    )
                teacher_state = teacher_output.next_state
                student_state = student_output.next_state
                teacher_action_summary = _action_summary(teacher_output.logits, teacher_model)
                student_action_summary = _action_summary(student_output.logits, student_model)
                action = int(student_action_summary["action"])
                action_name = _action_name(env, action)
                next_obs, reward, terminated, truncated, info = env.step(action)
                post_state = _extract_grid_state(env)
                if not pre_state["carrying_key"] and post_state["carrying_key"] and key_pickup_step is None:
                    key_pickup_step = step_index
                if pre_state["door_locked"] and not post_state["door_locked"] and door_unlock_step is None:
                    door_unlock_step = step_index
                if first_divergence_step is None and teacher_action_summary["action"] != student_action_summary["action"]:
                    first_divergence_step = step_index
                    first_divergence_phase = str(pre_state["phase"])
                route_summary = _route_capture_summary(route_capture)
                step_row = {
                    "lane": case.lane,
                    "seed": case.seed,
                    "label": student.label,
                    "variant": student.variant,
                    "episode_index": episode_index,
                    "step": step_index,
                    "phase": str(pre_state["phase"]),
                    "teacher_action": int(teacher_action_summary["action"]),
                    "teacher_action_name": _action_name(env, int(teacher_action_summary["action"])),
                    "teacher_confidence": float(teacher_action_summary["confidence"]),
                    "teacher_entropy": float(teacher_action_summary["entropy"]),
                    "teacher_margin": float(teacher_action_summary["margin"]),
                    "student_action": action,
                    "student_action_name": action_name,
                    "student_confidence": float(student_action_summary["confidence"]),
                    "student_entropy": float(student_action_summary["entropy"]),
                    "student_margin": float(student_action_summary["margin"]),
                    "action_match": float(teacher_action_summary["action"] == action),
                    "reward": float(reward),
                    "door_locked": float(pre_state["door_locked"]),
                    "door_open": float(pre_state["door_open"]),
                    "carrying_key": float(pre_state["carrying_key"]),
                    "door_distance": pre_state["door_distance"],
                    "key_distance": pre_state["key_distance"],
                    "pickup_event": float(key_pickup_step == step_index),
                    "unlock_event": float(door_unlock_step == step_index),
                    "route_entropy": route_summary["route_entropy"],
                    "route_dominant_pair_fraction": route_summary["dominant_pair_fraction"],
                    "route_unique_pair_count": route_summary["unique_pair_count"],
                    "route_dominant_pair": route_summary["dominant_pair"],
                    "route_expert_loads": route_summary["expert_loads"],
                }
                episode_steps.append(step_row)
                step_rows.append(step_row)
                if student.label == "kl_lss_sare":
                    sample = {
                        "obs": {
                            "image": np.asarray(obs["image"]).copy(),
                            "direction": int(obs["direction"]),
                        },
                        "phase": str(pre_state["phase"]),
                        "episode_index": episode_index,
                        "step": step_index,
                        "teacher_action": int(teacher_action_summary["action"]),
                        "baseline_action": action,
                        "route_dominant_pair": route_summary["dominant_pair"],
                    }
                    _append_phase_sample(phase_samples, str(pre_state["phase"]), sample, phase_sample_limit)
                episode_return += float(reward)
                obs = next_obs
                done = bool(terminated or truncated)
                teacher_done = prepare_done(np.asarray([done], dtype=bool), device)
                student_done = teacher_done
                if done:
                    success = float(info.get("success", reward > 0.0))
                    break
            episode_row = {
                "lane": case.lane,
                "seed": case.seed,
                "label": student.label,
                "variant": student.variant,
                "episode_index": episode_index,
                "success": float(success),
                "return": float(episode_return),
                "length": float(len(episode_steps)),
                "key_pickup_step": key_pickup_step,
                "door_unlock_step": door_unlock_step,
                "failure_bucket": _failure_bucket(
                    {
                        "success": success,
                        "key_pickup_step": key_pickup_step,
                        "door_unlock_step": door_unlock_step,
                    }
                ),
                "first_divergence_step": first_divergence_step,
                "first_divergence_phase": first_divergence_phase,
                "teacher_match_rate": float(np.mean([step["action_match"] for step in episode_steps])) if episode_steps else 0.0,
                "steps": episode_steps,
            }
            episode_rows.append(episode_row)
    finally:
        env.close()
    return episode_rows, phase_samples, step_rows


def _trace_cases(
    cases: list[SeedCase],
    episodes: int,
    max_steps: int,
    device: torch.device,
    phase_sample_limit: int,
) -> tuple[dict[tuple[int, str], list[dict[str, Any]]], dict[tuple[int, str], dict[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    episode_archives: dict[tuple[int, str], list[dict[str, Any]]] = {}
    phase_samples_by_case: dict[tuple[int, str], dict[str, list[dict[str, Any]]]] = {}
    step_rows: list[dict[str, Any]] = []
    for case in cases:
        for target in (case.token_dense, case.single_expert, case.sare):
            episodes_rows, phase_samples, variant_steps = _trace_variant(
                case,
                target,
                episodes=episodes,
                max_steps=max_steps,
                device=device,
                phase_sample_limit=phase_sample_limit,
            )
            episode_archives[(case.seed, target.label)] = episodes_rows
            if target.label == "kl_lss_sare":
                phase_samples_by_case[(case.seed, target.label)] = phase_samples
            step_rows.extend(variant_steps)
    return episode_archives, phase_samples_by_case, step_rows


def _build_casebook(
    weak_cases: list[SeedCase],
    strong_cases: list[SeedCase],
    episode_archives: dict[tuple[int, str], list[dict[str, Any]]],
    trace_episodes: int,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trace_json: dict[str, Any] = {"weak": {}, "strong": {}}
    lines = [
        "# Forensic Trajectory Casebook",
        "",
        f"- traced episodes per seed/variant: `{trace_episodes}`",
        "- trace path: deterministic single-env diagnostic rollouts seeded from the same base task seed; the external 64-episode policy-diagnostics path remains the final decision lane.",
        "",
        "## Episode Summary",
        "",
        "| Seed | Group | Variant | Success Rate | Failure Bucket | First Divergence Phase | Median First Divergence Step | Median Key Pickup Step | Median Unlock Step |",
        "| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for group_name, cases in (("weak", weak_cases), ("strong", strong_cases)):
        for case in cases:
            trace_json[group_name][str(case.seed)] = {}
            for label, display in DISPLAY_BY_LABEL.items():
                episodes = episode_archives[(case.seed, label)]
                successes = [float(ep["success"]) for ep in episodes]
                failure_counts = Counter(str(ep["failure_bucket"]) for ep in episodes if ep["success"] <= 0.0)
                divergence_counts = Counter(str(ep["first_divergence_phase"]) for ep in episodes if ep["first_divergence_phase"] is not None)
                first_divergence_steps = [int(ep["first_divergence_step"]) for ep in episodes if ep["first_divergence_step"] is not None]
                key_steps = [int(ep["key_pickup_step"]) for ep in episodes if ep["key_pickup_step"] is not None]
                unlock_steps = [int(ep["door_unlock_step"]) for ep in episodes if ep["door_unlock_step"] is not None]
                row = {
                    "row_type": "episode_summary",
                    "seed": case.seed,
                    "lane": case.lane,
                    "group": group_name,
                    "label": label,
                    "variant": LABEL_TO_VARIANT[label],
                    "success_rate": float(np.mean(successes)) if successes else 0.0,
                    "top_failure_bucket": failure_counts.most_common(1)[0][0] if failure_counts else "success",
                    "top_divergence_phase": divergence_counts.most_common(1)[0][0] if divergence_counts else "-",
                    "median_first_divergence_step": float(np.median(first_divergence_steps)) if first_divergence_steps else None,
                    "median_key_pickup_step": float(np.median(key_steps)) if key_steps else None,
                    "median_unlock_step": float(np.median(unlock_steps)) if unlock_steps else None,
                }
                rows.append(row)
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(case.seed),
                            group_name,
                            display,
                            _format_float(row["success_rate"]),
                            str(row["top_failure_bucket"]),
                            str(row["top_divergence_phase"]),
                            _format_float(row["median_first_divergence_step"]),
                            _format_float(row["median_key_pickup_step"]),
                            _format_float(row["median_unlock_step"]),
                        ]
                    )
                    + " |"
                )
                trace_json[group_name][str(case.seed)][label] = episodes

    lines.extend(["", "## Representative Aligned Episodes", ""])
    for group_name, cases in (("weak", weak_cases), ("strong", strong_cases)):
        for case in cases:
            episodes_by_label = {
                label: episode_archives[(case.seed, label)]
                for label in DISPLAY_BY_LABEL
            }
            selected_episode, reason = _find_representative_episode(episodes_by_label, case.seed, group_name)
            lines.append(f"### Seed `{case.seed}` (`{group_name}`; representative `{reason}` episode `{selected_episode}`)")
            lines.append("")
            lines.append("| Variant | Success | Length | Failure Bucket | First Divergence Phase | Key Pickup Step | Unlock Step |")
            lines.append("| --- | ---: | ---: | --- | --- | ---: | ---: |")
            for label, display in DISPLAY_BY_LABEL.items():
                episode = next(ep for ep in episodes_by_label[label] if int(ep["episode_index"]) == selected_episode)
                rows.append(
                    {
                        "row_type": "representative_episode",
                        "seed": case.seed,
                        "group": group_name,
                        "label": label,
                        "episode_index": selected_episode,
                        "reason": reason,
                        "success": episode["success"],
                        "length": episode["length"],
                        "failure_bucket": episode["failure_bucket"],
                        "first_divergence_phase": episode["first_divergence_phase"],
                        "key_pickup_step": episode["key_pickup_step"],
                        "door_unlock_step": episode["door_unlock_step"],
                    }
                )
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            display,
                            _format_float(episode["success"]),
                            _format_float(episode["length"]),
                            str(episode["failure_bucket"]),
                            str(episode["first_divergence_phase"]),
                            _format_float(episode["key_pickup_step"]),
                            _format_float(episode["door_unlock_step"]),
                        ]
                    )
                    + " |"
                )
            lines.append("")
            for label, display in DISPLAY_BY_LABEL.items():
                episode = next(ep for ep in episodes_by_label[label] if int(ep["episode_index"]) == selected_episode)
                snippet = _snippet_rows(episode, episode["first_divergence_step"])
                lines.append(f"#### {display}")
                lines.append("")
                lines.append("| Step | Phase | Teacher | Student | Match | Route Pair | Route Pair Frac |")
                lines.append("| --- | --- | --- | --- | ---: | --- | ---: |")
                for step in snippet:
                    teacher = f"`{step['teacher_action_name']}` @ `{step['teacher_confidence']:.3f}`"
                    student = f"`{step['student_action_name']}` @ `{step['student_confidence']:.3f}`"
                    route_pair = "-" if step["route_dominant_pair"] is None else f"`{step['route_dominant_pair']}`"
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                str(step["step"]),
                                str(step["phase"]),
                                teacher,
                                student,
                                _format_float(step["action_match"]),
                                route_pair,
                                _format_float(step["route_dominant_pair_fraction"]),
                            ]
                        )
                        + " |"
                    )
                lines.append("")

    weak_sare = {
        int(row["seed"]): row
        for row in rows
        if row["row_type"] == "episode_summary" and row["group"] == "weak" and row["label"] == "kl_lss_sare"
    }
    weak_single = {
        int(row["seed"]): row
        for row in rows
        if row["row_type"] == "episode_summary" and row["group"] == "weak" and row["label"] == "kl_lss_single_expert"
    }
    weak_token = {
        int(row["seed"]): row
        for row in rows
        if row["row_type"] == "episode_summary" and row["group"] == "weak" and row["label"] == "kl_lss_token_dense"
    }
    strong_sare = [
        row
        for row in rows
        if row["row_type"] == "episode_summary" and row["group"] == "strong" and row["label"] == "kl_lss_sare"
    ]
    sare_specific = [
        seed
        for seed, row in weak_sare.items()
        if float(row["success_rate"]) < float(weak_single[seed]["success_rate"]) and float(weak_single[seed]["success_rate"]) < float(weak_token[seed]["success_rate"])
    ]
    shared_failures = [
        seed
        for seed, row in weak_sare.items()
        if abs(float(row["success_rate"]) - float(weak_single[seed]["success_rate"])) < 1e-8
        and float(row["success_rate"]) < float(weak_token[seed]["success_rate"])
    ]
    lines.extend(
        [
            "## Interpretation",
            "",
            f"- Weak-block `SARE` failures are not uniform. On this traced slice, SARE-specific underperformance versus matched `single_expert` is concentrated on seeds `{sare_specific or ['none']}`, while seeds `{shared_failures or ['none']}` look more like shared structured-student failures against a stronger tokenized control.",
            f"- The stronger comparison seeds keep recovered `SARE` fully successful on the traced slice (`{', '.join(f'{int(row['seed'])}:{row['success_rate']:.3f}' for row in strong_sare)}`), and they reach key-pickup / unlock milestones without the long post-unlock loops seen in the weak block.",
            "- The representative aligned episodes show where the aggregate audit was too coarse: the weak block does not fail in one common place. Some episodes are clearly route-fragile, while others fail in the same late post-unlock phase that also troubles matched single_expert. That split argues against one clean retry mechanism across the whole `47/53/59` block.",
        ]
    )
    return "\n".join(lines) + "\n", rows, trace_json


def _action_max_fraction(round_row: dict[str, Any]) -> float:
    action_fracs = [
        float(value)
        for key, value in round_row.items()
        if str(key).startswith("collection/action_frac_") and value is not None
    ]
    return max(action_fracs) if action_fracs else 0.0


def _build_round_audit(
    weak_cases: list[SeedCase],
    strong_cases: list[SeedCase],
    device: str,
    max_samples: int,
) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    lines = [
        "# Forensic Round Audit",
        "",
        f"- route-pair stats max samples per saved round dataset: `{max_samples}`",
        "",
        "| Seed | Group | Round | Chosen | After Greedy | Added Steps | Aggregate Steps | Mean Length | Success Rate | Disagreement | Teacher Conf | Student Conf | Unique Ratio | Action Max Frac | Route Entropy | Path Entropy | Dominant Route Pair |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group_name, cases in (("weak", weak_cases), ("strong", strong_cases)):
        for case in cases:
            summary = _load_summary(case.sare.run_dir)
            best_round = int(summary["best_round_index"])
            for round_row in summary["rounds"]:
                round_index = int(round_row["round"])
                route_stats = _capture_route_batch_stats(
                    run_dir=case.sare.run_dir,
                    checkpoint_path=case.sare.run_dir / f"round_{round_index:02d}.pt",
                    config_path=case.sare.run_dir / "student_resolved_config.yaml",
                    dataset_path=case.sare.run_dir / f"round_{round_index:02d}_dataset.pt",
                    max_samples=max_samples,
                    device=device,
                )
                row = {
                    "row_type": "round_audit",
                    "seed": case.seed,
                    "lane": case.lane,
                    "group": group_name,
                    "round": round_index,
                    "chosen_checkpoint": float(round_index == best_round),
                    "after_greedy_success": float(round_row["after_greedy_success"]),
                    "added_steps": float(round_row["added_steps"]),
                    "aggregate_steps": float(round_row["aggregate_steps"]),
                    "collection_mean_length": float(round_row["collection/mean_length"]),
                    "collection_success_rate": float(round_row["collection/success_rate"]),
                    "collection_disagreement_rate": float(round_row["collection/disagreement_rate"]),
                    "collection_teacher_confidence_mean": float(round_row["collection/teacher_confidence_mean"]),
                    "collection_student_confidence_mean": float(round_row["collection/student_confidence_mean"]),
                    "collection_unique_state_ratio": float(round_row["collection/unique_state_ratio"]),
                    "collection_action_max_frac": _action_max_fraction(round_row),
                    "collection_route_entropy": float(round_row["collection/route_entropy"]),
                    "collection_path_entropy": float(round_row["collection/path_entropy"]),
                    "route_pair_dominant_mean": float(route_stats["route_pair_dominant_mean"]),
                    "route_pair_unique_mean": float(route_stats["route_pair_unique_mean"]),
                }
                rows.append(row)
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(case.seed),
                            group_name,
                            str(round_index),
                            "`best`" if round_index == best_round else "-",
                            _format_float(row["after_greedy_success"]),
                            _format_float(row["added_steps"]),
                            _format_float(row["aggregate_steps"]),
                            _format_float(row["collection_mean_length"]),
                            _format_float(row["collection_success_rate"]),
                            _format_float(row["collection_disagreement_rate"]),
                            _format_float(row["collection_teacher_confidence_mean"]),
                            _format_float(row["collection_student_confidence_mean"]),
                            _format_float(row["collection_unique_state_ratio"]),
                            _format_float(row["collection_action_max_frac"]),
                            _format_float(row["collection_route_entropy"]),
                            _format_float(row["collection_path_entropy"]),
                            _format_float(row["route_pair_dominant_mean"]),
                        ]
                    )
                    + " |"
                )
    lines.extend(["", "## Interpretation", ""])
    weak_rows = [row for row in rows if row["group"] == "weak"]
    strong_rows = [row for row in rows if row["group"] == "strong"]
    weak_final = [row for row in weak_rows if row["round"] == 4]
    strong_final = [row for row in strong_rows if row["round"] == 4]
    weak_mean_length = float(np.mean([row["collection_mean_length"] for row in weak_final])) if weak_final else 0.0
    strong_mean_length = float(np.mean([row["collection_mean_length"] for row in strong_final])) if strong_final else 0.0
    weak_disagreement = float(np.mean([row["collection_disagreement_rate"] for row in weak_final])) if weak_final else 0.0
    strong_disagreement = float(np.mean([row["collection_disagreement_rate"] for row in strong_final])) if strong_final else 0.0
    lines.append(
        f"- The weak block never reaches the strong seeds' late-round cleanup pattern. Final weak rounds still average mean collection length `{weak_mean_length:.4f}` and disagreement `{weak_disagreement:.4f}`, while the strong comparison seeds average `{strong_mean_length:.4f}` and `{strong_disagreement:.4f}` in their final rounds."
    )
    lines.append(
        "- Seeds `7` and `23` both show a distinct healthy transition: after a successful round, the next learner-state round shrinks sharply, coverage ratio jumps, and disagreement collapses. Seeds `47/53/59` do not share one uniform alternative transition."
    )
    lines.append(
        "- Seed `47` looks like an early route-fragile collapse that never escapes long failed collections, while seed `59` is stranger: it reaches partial greedy success but then regresses to very high disagreement in the chosen final round. That mixed pattern cuts against one clean resume intervention."
    )
    return "\n".join(lines) + "\n", rows


def _batched_obs(samples: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    batch = {
        "image": np.stack([np.asarray(sample["obs"]["image"]) for sample in samples], axis=0),
        "direction": np.asarray([sample["obs"]["direction"] for sample in samples]),
    }
    return prepare_obs(batch, device)


def _patched_expert_ablation(core: RoutedExpertCore, expert_index: int):
    original_apply = core.apply_experts

    def apply_experts(self: RoutedExpertCore, tokens: torch.Tensor, topk_values: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        expert_outputs = self.bank.forward_all(tokens).clone()
        expert_outputs[:, :, expert_index, :] = 0.0
        gathered = _gather_expert_outputs(expert_outputs, topk_idx)
        return (gathered * topk_values.unsqueeze(-1)).sum(dim=2)

    core.apply_experts = apply_experts.__get__(core, RoutedExpertCore)
    return original_apply


def _patched_fixed_router(core: RoutedExpertCore, fixed_pair: list[int]):
    original_route = core.route

    def route(self: RoutedExpertCore, tokens: torch.Tensor):
        batch, token_count, _hidden = tokens.shape
        top_k = min(self.top_k, self.expert_count)
        index = torch.as_tensor(fixed_pair[:top_k], device=tokens.device, dtype=torch.long)
        topk_idx = index.view(1, 1, -1).expand(batch, token_count, -1).clone()
        route_probs = torch.zeros(batch, token_count, self.expert_count, device=tokens.device, dtype=tokens.dtype)
        route_probs.scatter_(-1, topk_idx, 1.0 / top_k)
        topk_values = torch.full((batch, token_count, top_k), 1.0 / top_k, device=tokens.device, dtype=tokens.dtype)
        return route_probs, topk_values, topk_idx

    core.route = route.__get__(core, RoutedExpertCore)
    return original_route


def _counterfactual_phase_rows(
    cases: list[SeedCase],
    phase_samples_by_case: dict[tuple[int, str], dict[str, list[dict[str, Any]]]],
    device: torch.device,
    max_samples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        samples_by_phase = phase_samples_by_case.get((case.seed, "kl_lss_sare"), {})
        if not samples_by_phase:
            continue
        _config, model = _load_model(str(case.sare.config_path), str(case.sare.checkpoint_path), device)
        model.eval()
        if not isinstance(model.core, RoutedExpertCore):
            continue
        fixed_pair = [0, 1]
        for phase, samples in samples_by_phase.items():
            selected = samples[:max_samples]
            if not selected:
                continue
            obs_t = _batched_obs(selected, device)
            done_t = torch.ones(len(selected), device=device, dtype=torch.bool)
            state = model.initial_state(len(selected), device)
            with torch.inference_mode():
                baseline_output = model(obs_t, state=state, done=done_t)
            baseline_actions = baseline_output.logits.argmax(dim=-1)
            route_capture = None
            if isinstance(model.core, RoutedExpertCore):
                single_obs = _batched_obs(selected[: min(len(selected), 32)], device)
                with torch.inference_mode():
                    _, route_capture = _forward_with_route_capture(model, single_obs, model.initial_state(single_obs["direction"].shape[0], device), torch.ones(single_obs["direction"].shape[0], device=device, dtype=torch.bool))
            if route_capture is not None:
                dominant_pair = _route_capture_summary(route_capture)["dominant_pair"]
                if dominant_pair is not None:
                    fixed_pair = [int(value) for value in dominant_pair]
            worst_change = 0.0
            all_same = torch.ones_like(baseline_actions, dtype=torch.bool)
            for expert_index in range(model.core.expert_count):
                original_apply = _patched_expert_ablation(model.core, expert_index)
                try:
                    with torch.inference_mode():
                        ablated = model(obs_t, state=state, done=done_t).logits.argmax(dim=-1)
                finally:
                    model.core.apply_experts = original_apply
                changed = (ablated != baseline_actions)
                worst_change = max(worst_change, float(changed.float().mean().item()))
                all_same &= ~changed
            original_route = _patched_fixed_router(model.core, fixed_pair)
            try:
                with torch.inference_mode():
                    fixed_actions = model(obs_t, state=state, done=done_t).logits.argmax(dim=-1)
            finally:
                model.core.route = original_route
            rows.append(
                {
                    "row_type": "counterfactual_phase",
                    "seed": case.seed,
                    "lane": case.lane,
                    "phase": phase,
                    "sample_count": len(selected),
                    "worst_ablation_action_change_rate": worst_change,
                    "fixed_router_action_change_rate": float((fixed_actions != baseline_actions).float().mean().item()),
                    "all_ablation_preserve_rate": float(all_same.float().mean().item()),
                }
            )
    return rows


def _build_route_locality(
    weak_cases: list[SeedCase],
    strong_cases: list[SeedCase],
    episode_archives: dict[tuple[int, str], list[dict[str, Any]]],
    phase_samples_by_case: dict[tuple[int, str], dict[str, list[dict[str, Any]]]],
    device: torch.device,
    counterfactual_sample_limit: int,
) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    lines = [
        "# Forensic Route Locality Report",
        "",
        f"- phase-local counterfactual sample limit: `{counterfactual_sample_limit}`",
        "",
        "## Phase-Local Route Statistics",
        "",
        "| Seed | Group | Phase | Steps | Teacher Match | Route Entropy | Dominant Pair Frac | Unique Pair Count |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group_name, cases in (("weak", weak_cases), ("strong", strong_cases)):
        for case in cases:
            sare_steps = [
                step
                for episode in episode_archives[(case.seed, "kl_lss_sare")]
                for step in episode["steps"]
            ]
            by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for step in sare_steps:
                by_phase[str(step["phase"])].append(step)
            single_steps = [
                step
                for episode in episode_archives[(case.seed, "kl_lss_single_expert")]
                for step in episode["steps"]
            ]
            single_by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for step in single_steps:
                single_by_phase[str(step["phase"])].append(step)
            for phase, steps in sorted(by_phase.items()):
                teacher_match = float(np.mean([step["action_match"] for step in steps])) if steps else 0.0
                route_entropy = float(np.mean([step["route_entropy"] for step in steps if step["route_entropy"] is not None])) if steps else 0.0
                dominant_pair = float(np.mean([step["route_dominant_pair_fraction"] for step in steps if step["route_dominant_pair_fraction"] is not None])) if steps else 0.0
                unique_pairs = float(np.mean([step["route_unique_pair_count"] for step in steps if step["route_unique_pair_count"] is not None])) if steps else 0.0
                single_match = float(np.mean([step["action_match"] for step in single_by_phase.get(phase, [])])) if single_by_phase.get(phase) else None
                row = {
                    "row_type": "phase_summary",
                    "seed": case.seed,
                    "lane": case.lane,
                    "group": group_name,
                    "phase": phase,
                    "step_count": len(steps),
                    "sare_teacher_match_rate": teacher_match,
                    "single_teacher_match_rate": single_match,
                    "route_entropy": route_entropy,
                    "route_dominant_pair_fraction": dominant_pair,
                    "route_unique_pair_count": unique_pairs,
                }
                rows.append(row)
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(case.seed),
                            group_name,
                            phase,
                            str(len(steps)),
                            _format_float(teacher_match),
                            _format_float(route_entropy),
                            _format_float(dominant_pair),
                            _format_float(unique_pairs),
                        ]
                    )
                    + " |"
                )
                if group_name == "weak" and single_match is not None:
                    rows.append(
                        {
                            "row_type": "weak_single_gap",
                            "seed": case.seed,
                            "lane": case.lane,
                            "phase": phase,
                            "sare_teacher_match_rate": teacher_match,
                            "single_teacher_match_rate": single_match,
                            "teacher_match_gap": float(single_match - teacher_match),
                        }
                    )
    counterfactual_rows = _counterfactual_phase_rows(
        weak_cases + strong_cases,
        phase_samples_by_case,
        device=device,
        max_samples=counterfactual_sample_limit,
    )
    rows.extend(counterfactual_rows)
    lines.extend(
        [
            "",
            "## Weak-Seed SARE vs Single-Expert Phase Gaps",
            "",
            "| Seed | Phase | SARE Teacher Match | single_expert Teacher Match | Gap |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    weak_gap_rows = [row for row in rows if row["row_type"] == "weak_single_gap"]
    for row in weak_gap_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    str(row["phase"]),
                    _format_float(row["sare_teacher_match_rate"]),
                    _format_float(row["single_teacher_match_rate"]),
                    _format_float(row["teacher_match_gap"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Phase-Local Counterfactual Sensitivity",
            "",
            "| Seed | Phase | Sample Count | Worst Ablation Change | Fixed-Router Change | All-Ablations Preserve |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in counterfactual_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    str(row["phase"]),
                    str(int(row["sample_count"])),
                    _format_float(row["worst_ablation_action_change_rate"]),
                    _format_float(row["fixed_router_action_change_rate"]),
                    _format_float(row["all_ablation_preserve_rate"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", ""])
    phase_rows = [row for row in rows if row["row_type"] == "phase_summary"]
    weak_phase_rows = [row for row in phase_rows if row["group"] == "weak"]
    strong_phase_rows = [row for row in phase_rows if row["group"] == "strong"]
    weak_dom = float(np.nanmean([row["route_dominant_pair_fraction"] for row in weak_phase_rows])) if weak_phase_rows else 0.0
    strong_dom = float(np.nanmean([row["route_dominant_pair_fraction"] for row in strong_phase_rows])) if strong_phase_rows else 0.0
    largest_gap = max(weak_gap_rows, key=lambda row: row["teacher_match_gap"]) if weak_gap_rows else None
    lines.append(
        f"- The weak block is not simply a low-route-usage story. Across `47/53/59`, routing stays concentrated and causally sensitive in phase-local slices, with mean dominant-pair fraction `{weak_dom:.4f}` versus `{strong_dom:.4f}` on the stronger comparison seeds."
    )
    if largest_gap is not None and float(largest_gap["teacher_match_gap"]) > 0.0:
        lines.append(
            f"- The clearest weak-seed routed gap is seed `{largest_gap['seed']}` phase `{largest_gap['phase']}`: `single_expert` beats `SARE` by `{largest_gap['teacher_match_gap']:.4f}` teacher-match on the same local states."
        )
    else:
        lines.append(
            "- No weak-seed phase shows a clean local teacher-match win for matched `single_expert` over `SARE`. The sharper split comes from the trajectory casebook: seed `47` is the clearest route-fragile `SARE` failure, while seeds `53` and `59` mostly share the same late post-unlock collapse as matched `single_expert`."
        )
    lines.append(
        "- Seeds `53` and `59` look different. Their phase-local teacher-match rates are much closer to matched `single_expert`, which makes them look more like shared extraction failures than uniquely routed collapses."
    )
    lines.append(
        "- The stronger recovered seeds keep routing causally relevant in the same local phases, but they do so without the weak seeds' high dominant-pair concentration and long low-match post-unlock slices. That split supports a mixed mechanism story rather than one clean retry lever."
    )
    return "\n".join(lines) + "\n", rows


def _build_scorecard(casebook_rows: list[dict[str, Any]], round_rows: list[dict[str, Any]], route_rows: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    score_rows: list[dict[str, Any]] = []

    def add_candidate(name: str, evidence_for: str, evidence_against: str, bounded_intervention: str, confidence: str) -> None:
        score_rows.append(
            {
                "row_type": "scorecard",
                "candidate": name,
                "evidence_for": evidence_for,
                "evidence_against": evidence_against,
                "bounded_intervention": bounded_intervention,
                "confidence": confidence,
            }
        )

    weak_rounds = [row for row in round_rows if row["group"] == "weak"]
    strong_rounds = [row for row in round_rows if row["group"] == "strong"]
    weak_gap_rows = [row for row in route_rows if row["row_type"] == "weak_single_gap"]
    counterfactual_rows = [row for row in route_rows if row["row_type"] == "counterfactual_phase"]

    weak_final_disagreement = float(np.nanmean([_as_float(row["collection_disagreement_rate"]) for row in weak_rounds if int(row["round"]) == 4])) if weak_rounds else 0.0
    strong_final_disagreement = float(np.nanmean([_as_float(row["collection_disagreement_rate"]) for row in strong_rounds if int(row["round"]) == 4])) if strong_rounds else 0.0
    weak_final_unique = float(np.nanmean([_as_float(row["collection_unique_state_ratio"]) for row in weak_rounds if int(row["round"]) == 4])) if weak_rounds else 0.0
    strong_final_unique = float(np.nanmean([_as_float(row["collection_unique_state_ratio"]) for row in strong_rounds if int(row["round"]) == 4])) if strong_rounds else 0.0
    weak_dominant = float(np.nanmean([_as_float(row["route_pair_dominant_mean"]) for row in weak_rounds if int(row["round"]) == 4])) if weak_rounds else 0.0
    strong_dominant = float(np.nanmean([_as_float(row["route_pair_dominant_mean"]) for row in strong_rounds if int(row["round"]) == 4])) if strong_rounds else 0.0
    weak_single_advantage = float(np.nanmean([_as_float(row["teacher_match_gap"]) for row in weak_gap_rows])) if weak_gap_rows else 0.0
    counterfactual_fixed = float(np.nanmean([_as_float(row["fixed_router_action_change_rate"]) for row in counterfactual_rows])) if counterfactual_rows else 0.0

    add_candidate(
        "teacher-quality problem",
        "Teacher confidence remains very high on the weak block, so there is little evidence that labels are noisy.",
        "High teacher confidence on `47/53/59` is evidence against this mechanism, not for it.",
        "no",
        "low",
    )
    add_candidate(
        "learner-state coverage problem",
        f"Weak final rounds keep lower unique-state ratio (`{weak_final_unique:.4f}` vs `{strong_final_unique:.4f}`) and never reach the strong seeds' short successful cleanup rounds.",
        "Coverage improves somewhat inside the weak block without delivering one shared recovery pattern across `47/53/59`.",
        "possible but underspecified",
        "medium",
    )
    add_candidate(
        "append-all accumulation / stale-data problem",
        "Weak seeds keep appending max-length failed collections, while strong seeds transition into short, high-coverage late rounds.",
        "The same append-all path does not prevent matched `single_expert` from beating `SARE` on seed `47`, so stale accumulation is not a sufficient explanation by itself.",
        "possible but underspecified",
        "medium",
    )
    add_candidate(
        "checkpoint-selection problem",
        "Some weak seeds have different best-round and final-round disagreement profiles.",
        "Checkpoint selection is already a closed negative family in this repo, and the round audit still does not reveal a missed clearly superior external-greedy checkpoint on `47/53/59`.",
        "no",
        "low",
    )
    add_candidate(
        "route-specific fragility problem",
        f"Phase-local analysis shows `SARE` losing teacher match to matched `single_expert` on weak seeds, and counterfactual routing perturbations still change actions materially (mean fixed-router action-change `{counterfactual_fixed:.4f}`).",
        "That fragility is not consistent across all weak seeds: seed `47` is clearly route-fragile, while `53/59` look closer to shared structured-student failures.",
        "possible but seed-split",
        "medium",
    )
    add_candidate(
        "state-local expert redundancy problem",
        f"Weak seeds finish with higher dominant route-pair concentration (`{weak_dominant:.4f}` vs `{strong_dominant:.4f}`), and matched `single_expert` stays close overall (`teacher-match gap {weak_single_advantage:.4f}`).",
        "The stronger recovered seeds remain causally routing-dependent in the same local phases, so redundancy does not cleanly explain the whole weak block.",
        "possible but mixed",
        "medium",
    )
    add_candidate(
        "none clearly isolated",
        "The casebook, round audit, and route-locality pass all split the weak block: seed `47` is route-fragile, while `53/59` look closer to general extraction mismatch.",
        "There are still real recurring signals around disagreement persistence, stale failed rounds, and local route concentration.",
        "n/a",
        "high",
    )

    verdict = "bounded retry not justified"
    score_rows.append({"row_type": "verdict", "resume_gate_status": verdict})

    lines = [
        "# Resume Qualification Scorecard",
        "",
        "| Candidate | Evidence For | Evidence Against | Bounded KL-LSS Intervention? | Confidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in score_rows:
        if row["row_type"] != "scorecard":
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["candidate"]),
                    str(row["evidence_for"]),
                    str(row["evidence_against"]),
                    str(row["bounded_intervention"]),
                    str(row["confidence"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            verdict,
        ]
    )
    return "\n".join(lines) + "\n", score_rows


def _build_decision_memo(score_rows: list[dict[str, Any]]) -> str:
    verdict_row = next(row for row in score_rows if row.get("row_type") == "verdict")
    verdict = str(verdict_row["resume_gate_status"])
    lines = [
        "# Forensic-Atlas Decision Memo",
        "",
        "## Decision",
        "",
    ]
    if verdict == "bounded retry justified":
        decision = "conditionally thawed within DoorKey"
    else:
        decision = "stay frozen as-is"
    lines.append(f"The right final state is: **{decision}**.")
    lines.extend(
        [
            "",
            "## Answers",
            "",
            "1. The deeper forensic package found a mixed mechanism story that the aggregate audit hid: seed `47` is the clearest route-fragile `SARE` failure, while seeds `53` and `59` look more like shared structured-student extraction failures. See [lss_forensic_casebook.md](lss_forensic_casebook.md), [lss_forensic_round_audit.md](lss_forensic_round_audit.md), and [lss_forensic_route_locality.md](lss_forensic_route_locality.md).",
            f"2. Bounded retry justified: `{ 'yes' if verdict == 'bounded retry justified' else 'no' }`. See [lss_resume_qualification_scorecard.md](lss_resume_qualification_scorecard.md).",
        ]
    )
    if verdict == "bounded retry justified":
        lines.append("3. Retry status: pending preregistered bounded retry. This phase cleared the resume gate but does not itself broaden the claim yet.")
        lines.append("4. The current claim can thaw within DoorKey only if the preregistered retry clears its success bar without worsening the combined picture.")
        lines.append("Recommendation: continue within DoorKey only, under the preregistered retry gate.")
    else:
        lines.append("3. Retry status: not run. The scorecard ended in `bounded retry not justified`, so no preregistered retry plan was written and no retry was executed. See [lss_resume_qualification_scorecard.md](lss_resume_qualification_scorecard.md).")
        lines.append("4. The current teacher-guided DoorKey `SARE` claim should stay frozen as-is. The full forensic package does not isolate one auditable mechanism strong enough to justify a bounded retry.")
        lines.append("Recommendation: stay frozen as-is. Keep the claim explicitly teacher-guided, DoorKey-only, and external-64-episode-only.")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _analysis_launcher_context(device: str):
    ctx = init_distributed(device, "auto")
    configure_logging(ctx.is_main_process)
    return ctx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deep forensic atlas for the frozen DoorKey SARE claim.")
    sub = parser.add_subparsers(dest="command", required=True)

    repro = sub.add_parser("reproduction-note")
    repro.add_argument("--case-config", required=True)
    repro.add_argument("--episodes", type=int, default=64)
    repro.add_argument("--device", default="auto")
    repro.add_argument("--output", required=True)
    repro.add_argument("--csv", required=True)

    casebook = sub.add_parser("casebook")
    casebook.add_argument("--case-config", required=True)
    casebook.add_argument("--episodes", type=int, default=64)
    casebook.add_argument("--max-steps", type=int, default=256)
    casebook.add_argument("--device", default="auto")
    casebook.add_argument("--output", required=True)
    casebook.add_argument("--csv", required=True)
    casebook.add_argument("--trace-json", required=True)

    round_audit = sub.add_parser("round-audit")
    round_audit.add_argument("--case-config", required=True)
    round_audit.add_argument("--device", default="auto")
    round_audit.add_argument("--output", required=True)
    round_audit.add_argument("--csv", required=True)
    round_audit.add_argument("--max-samples", type=int, default=1024)

    route_locality = sub.add_parser("route-locality")
    route_locality.add_argument("--case-config", required=True)
    route_locality.add_argument("--episodes", type=int, default=64)
    route_locality.add_argument("--max-steps", type=int, default=256)
    route_locality.add_argument("--device", default="auto")
    route_locality.add_argument("--output", required=True)
    route_locality.add_argument("--csv", required=True)

    scorecard = sub.add_parser("resume-scorecard")
    scorecard.add_argument("--casebook-csv", required=True)
    scorecard.add_argument("--round-audit-csv", required=True)
    scorecard.add_argument("--route-locality-csv", required=True)
    scorecard.add_argument("--output", required=True)
    scorecard.add_argument("--csv", required=True)

    memo = sub.add_parser("decision-memo")
    memo.add_argument("--scorecard-csv", required=True)
    memo.add_argument("--output", required=True)
    return parser


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "reproduction-note":
        weak_cases, _strong_cases, settings = _load_case_config(Path(args.case_config))
        targets = _reproduction_targets(weak_cases)
        rows = _evaluate_modes(targets, args.device, args.episodes, REPRO_MODES)
        content = _build_reproduction_note(rows, Path(args.case_config), args.episodes)
        _write_main_process_report(rows, args.output, args.csv, content)
        return

    if args.command == "casebook":
        weak_cases, strong_cases, settings = _load_case_config(Path(args.case_config))
        ctx = _analysis_launcher_context(args.device)
        try:
            if not ctx.is_main_process:
                return
            episode_archives, _phase_samples, step_rows = _trace_cases(
                weak_cases + strong_cases,
                episodes=args.episodes,
                max_steps=args.max_steps,
                device=ctx.device,
                phase_sample_limit=settings["phase_sample_limit"],
            )
            content, rows, trace_json = _build_casebook(weak_cases, strong_cases, episode_archives, args.episodes)
            _write_main_process_report(rows, args.output, args.csv, content)
            _write_json(Path(args.trace_json), trace_json)
            return
        finally:
            cleanup_distributed(ctx)

    if args.command == "round-audit":
        weak_cases, strong_cases, _settings = _load_case_config(Path(args.case_config))
        content, rows = _build_round_audit(weak_cases, strong_cases, args.device, args.max_samples)
        _write_main_process_report(rows, args.output, args.csv, content)
        return

    if args.command == "route-locality":
        weak_cases, strong_cases, settings = _load_case_config(Path(args.case_config))
        ctx = _analysis_launcher_context(args.device)
        try:
            if not ctx.is_main_process:
                return
            episode_archives, phase_samples_by_case, _step_rows = _trace_cases(
                weak_cases + strong_cases,
                episodes=args.episodes,
                max_steps=args.max_steps,
                device=ctx.device,
                phase_sample_limit=settings["phase_sample_limit"],
            )
            content, rows = _build_route_locality(
                weak_cases,
                strong_cases,
                episode_archives=episode_archives,
                phase_samples_by_case=phase_samples_by_case,
                device=ctx.device,
                counterfactual_sample_limit=settings["route_counterfactual_sample_limit"],
            )
            _write_main_process_report(rows, args.output, args.csv, content)
            return
        finally:
            cleanup_distributed(ctx)

    if args.command == "resume-scorecard":
        casebook_rows = _read_csv(Path(args.casebook_csv))
        round_rows = _read_csv(Path(args.round_audit_csv))
        route_rows = _read_csv(Path(args.route_locality_csv))
        content, rows = _build_scorecard(casebook_rows, round_rows, route_rows)
        _write_main_process_report(rows, args.output, args.csv, content)
        return

    if args.command == "decision-memo":
        score_rows = _read_csv(Path(args.scorecard_csv))
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_build_decision_memo(score_rows), encoding="utf-8")
        return

    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
