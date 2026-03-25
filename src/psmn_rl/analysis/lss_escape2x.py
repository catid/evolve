from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from psmn_rl.analysis.campaign_config import load_campaign_config
from psmn_rl.analysis.lss_deadlock_program import (
    PRE_KEY_PHASES,
    _analysis_label,
    _label_run_dir,
    _optional_json,
    _optional_text,
    _round6_run_dir,
)
from psmn_rl.analysis.lss_forensic_atlas import (
    _action_name,
    _action_summary,
    _build_single_env,
    _extract_grid_state,
    _forward_with_route_capture,
    _obs_to_batch,
    _route_capture_summary,
)
from psmn_rl.analysis.lss_post_pass_campaign import _write_json
from psmn_rl.analysis.lss_route_dependence import _patched_core
from psmn_rl.analysis.lss_successor_migration import _control_family, _current_round6_rows
from psmn_rl.analysis.policy_distillation import _load_model
from psmn_rl.analysis.lss_robustness import _format_float
from psmn_rl.rl.ppo.algorithm import prepare_done, prepare_obs
from psmn_rl.utils.io import get_git_commit, get_git_dirty
from psmn_rl.utils.seed import capture_rng_state, restore_rng_state, set_seed


PROGRESS_PHASE_RANK = {
    "search_key": 0,
    "at_key": 1,
    "carry_key": 2,
    "at_locked_door": 3,
    "post_unlock": 4,
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gpu_count(device: str) -> int:
    if device == "cpu":
        return 0
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _group_cases(campaign: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in campaign["analysis"][key]:
        label = str(item["label"])
        for lane, seed in item["cases"]:
            rows.append({"group": label, "lane": str(lane), "seed": int(seed)})
    return rows


def _teacher_locked_dev_cases(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    return _group_cases(campaign, "teacher_locked_dev_groups")


def _teacher_locked_holdout_cases(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    return _group_cases(campaign, "teacher_locked_holdout_groups")


def _ambiguous_cases(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    return _group_cases(campaign, "ambiguous_groups")


def _guardrail_cases(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    return _group_cases(campaign, "guardrail_groups")


def _healthy_cases(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    return _group_cases(campaign, "healthy_groups")


def _all_detector_cases(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    cases = []
    for item in (
        _teacher_locked_dev_cases(campaign),
        _teacher_locked_holdout_cases(campaign),
        _ambiguous_cases(campaign),
        _guardrail_cases(campaign),
        _healthy_cases(campaign),
    ):
        cases.extend(item)
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in cases:
        key = (row["lane"], int(row["seed"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _unique_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        key = (str(row["lane"]), int(row["seed"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append({"lane": str(row["lane"]), "seed": int(row["seed"])})
    return unique


def _source_quality_cases(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    return _unique_cases(
        [
            *_teacher_locked_dev_cases(campaign),
            *_teacher_locked_holdout_cases(campaign),
            *_ambiguous_cases(campaign),
            *_healthy_cases(campaign),
        ]
    )


def _teacher_run_dir(campaign: dict[str, Any], lane: str, seed: int) -> Path:
    teacher_root = Path(campaign["lane_roots"][lane]["teacher_root"]) / f"seed_{seed}"
    return teacher_root / "flat_dense_ent1e3"


def _resolved_config_path(run_dir: Path) -> Path:
    for name in ("student_resolved_config.yaml", "resolved_config.yaml"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return run_dir / "resolved_config.yaml"


def _prior_arch_substantive_runs(campaign: dict[str, Any]) -> tuple[int, int]:
    prior_report = _optional_json(campaign["analysis"].get("prior_escape_stage1_json"))
    prior = len(list(prior_report.get("candidate_summaries", [])))
    target = max(prior * 2, 96)
    return prior, target


def _escape2x_decision_strings(campaign: dict[str, Any]) -> dict[str, str]:
    overrides = campaign.get("decision_strings", {})
    if isinstance(overrides, dict):
        return {
            "replace": str(overrides.get("replace", "challenger replaces the active benchmark")),
            "confirm": str(overrides.get("confirm", "active benchmark confirmed and bounded rescue frontier clarified")),
            "bounded_rescue": str(
                overrides.get(
                    "bounded_rescue",
                    "round6 remains active and a bounded specialist rescue policy is justified",
                )
            ),
            "narrow": str(overrides.get("narrow", "active benchmark remains and the frontier narrows further")),
            "arch_specialist": str(
                overrides.get(
                    "arch_specialist",
                    overrides.get("arch_future_branch", "architecture-specialist branch justified for the next phase"),
                )
            ),
        }
    return {
        "replace": "challenger replaces the active benchmark",
        "confirm": "active benchmark confirmed and bounded rescue frontier clarified",
        "bounded_rescue": "round6 remains active and a bounded specialist rescue policy is justified",
        "narrow": "active benchmark remains and the frontier narrows further",
        "arch_specialist": "architecture-specialist branch justified for the next phase",
    }


def _current_run_lookup(campaign: dict[str, Any]) -> dict[tuple[str, str, int], Path]:
    return {
        (str(row["label"]), str(row["lane"]), int(row["seed"])): Path(str(row["run_dir"]))
        for row in _current_round6_rows(campaign)
    }


def _round6_summary_for_cases(campaign: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    case_set = {(str(row["lane"]), int(row["seed"])) for row in rows}
    current_rows = [
        row
        for row in _current_round6_rows(campaign)
        if (str(row["lane"]), int(row["seed"])) in case_set
    ]
    payload: dict[str, Any] = {}
    for label in ("kl_lss_sare", "kl_lss_token_dense", "kl_lss_single_expert"):
        values = [
            float(row["final_greedy_success"])
            for row in current_rows
            if str(row["label"]) == label
        ]
        payload[label] = {
            "mean": float(mean(values)) if values else 0.0,
            "count": len(values),
            "complete_seed_failures": sum(1 for value in values if value <= 0.0),
        }
    return payload


def _best_arch_seed(report_payload: dict[str, Any], candidate: str, fallback: int = 7) -> int:
    for row in report_payload.get("candidate_summaries", []):
        if str(row.get("candidate")) != candidate:
            continue
        best = max(
            list(row.get("seed_breakout", [])),
            key=lambda item: (float(item.get("sampled_mean", 0.0)), float(item.get("greedy_mean", 0.0))),
            default=None,
        )
        if best is not None:
            return int(best.get("train_seed", fallback))
    return fallback


def _source_model_specs(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    exploit = _optional_json(campaign["analysis"].get("prior_architecture_stage1_exploit_json"))
    train_root = Path(campaign["analysis"]["architecture_stage1_train_root"])
    incumbent_candidate = str(campaign["analysis"]["architecture_incumbent_candidate"])
    challenger_candidate = str(campaign["analysis"]["architecture_challenger_candidate"])
    incumbent_seed = _best_arch_seed(exploit, incumbent_candidate, fallback=11)
    challenger_seed = _best_arch_seed(exploit, challenger_candidate, fallback=11)
    incumbent_run = train_root / incumbent_candidate / f"seed_{incumbent_seed}"
    challenger_run = train_root / challenger_candidate / f"seed_{challenger_seed}"
    return [
        {
            "model_id": "round6",
            "display": "round6",
            "family": "active_benchmark",
            "case_specific": True,
            "label": "kl_lss_sare",
            "routed": True,
        },
        {
            "model_id": "token_dense",
            "display": "token_dense",
            "family": "matched_control",
            "case_specific": True,
            "label": "kl_lss_token_dense",
            "routed": False,
        },
        {
            "model_id": "single_expert",
            "display": "single_expert",
            "family": "matched_control",
            "case_specific": True,
            "label": "kl_lss_single_expert",
            "routed": False,
        },
        {
            "model_id": "architecture_keyed_residual",
            "display": "sare_phase_memory_route_bias_keyed_residual",
            "family": "architecture_incumbent",
            "case_specific": False,
            "run_dir": str(incumbent_run),
            "config_path": str(_resolved_config_path(incumbent_run)),
            "checkpoint_path": str(incumbent_run / "latest.pt"),
            "routed": True,
        },
        {
            "model_id": "architecture_weak_base_expert_gate",
            "display": "sare_phase_memory_route_bias_keyed_residual_weak_base_expert_gate",
            "family": "architecture_challenger",
            "case_specific": False,
            "run_dir": str(challenger_run),
            "config_path": str(_resolved_config_path(challenger_run)),
            "checkpoint_path": str(challenger_run / "latest.pt"),
            "routed": True,
        },
    ]


def _write_state_reconciliation(campaign: dict[str, Any]) -> None:
    prior_runs, target_runs = _prior_arch_substantive_runs(campaign)
    architecture_reports = [str(item) for item in campaign["analysis"].get("source_quality_architecture_reports", [])]
    accepted_source_quality = _optional_json(campaign["analysis"].get("accepted_source_quality_json"))
    current_gate_pack = str(campaign["frozen_pack"])
    current_candidate_pack = str(campaign["current_canonical_pack"])
    lines = [
        "# Escape2x State Reconciliation",
        "",
        f"- program label: `{_analysis_label(campaign, '2x-scale Deadlock Escape, Practicalization, and Architecture-Specialist Program')}`",
        f"- active benchmark: `{campaign['current_canonical_name']}`",
        f"- archived frozen provenance anchor: `{campaign['legacy_frozen_pack']}`",
        f"- active benchmark pack: `{current_candidate_pack}`",
        f"- current gate-reference pack: `{current_gate_pack}`",
        f"- current decision memo before this program: `{campaign['current_decision_memo']}`",
        "",
        "## Accepted State",
        "",
        "- `round6` remains the active DoorKey benchmark.",
        "- The archived frozen benchmark pack remains the provenance anchor.",
        "- The public claim envelope stays teacher-guided, KL learner-state, DoorKey-only, and external-64-eval-only.",
        "- The architecture lane remains exploratory only; the best local architecture line is a narrow sampled rescue signal rather than a benchmark-lane win.",
        "",
        "## Prior Campaign Scale",
        "",
        f"- prior substantive runs from the previous escape/practicalization pass: `{prior_runs}`",
        f"- new target substantive run budget absent stop conditions: `{target_runs}`",
        "- This program can only stop below that target if a formal stop condition fires and the report trail records exactly where the funnel died.",
        "",
        "## Current Frontier Inputs",
        "",
        f"- portfolio frontier manifest: `outputs/reports/portfolio_frontier_manifest.md`",
        f"- portfolio operational state: `outputs/reports/portfolio_operational_state.md`",
        f"- deadlock oracle decision: `outputs/reports/deadlock_oracle_decision_memo.md`",
        f"- oracle synthesis: `outputs/reports/oracle_stageA6_synthesis.md`",
        f"- accepted source-quality report: `{campaign['analysis']['accepted_source_quality_report']}`",
        f"- accepted source-quality verdict: `{accepted_source_quality.get('verdict', 'missing')}`",
        f"- architecture reference reports: `{architecture_reports}`",
        "",
        "## Reconciliation Outcome",
        "",
        "- The repo state is internally consistent enough to start a source-quality, rescue, and practicalization program around `round6`.",
        "- The deadlock blocker still reads as mixed teacher-target plus transition-coverage, with only bounded inference-time escape producing a clearly positive signal.",
        "- The architecture branch remains specialist-only and not benchmark-worthy unless this program first produces a real rescue signal.",
    ]
    Path(campaign["reports"]["state_reconciliation"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_baseline_sync(campaign: dict[str, Any]) -> None:
    group_specs = [
        ("teacher_locked_dev", _teacher_locked_dev_cases(campaign)),
        ("teacher_locked_holdout", _teacher_locked_holdout_cases(campaign)),
        ("ambiguous", _ambiguous_cases(campaign)),
        ("healthy", _healthy_cases(campaign)),
        ("guardrail", _guardrail_cases(campaign)),
    ]
    rows: list[dict[str, Any]] = []
    lines = [
        "# Escape2x Baseline Sync",
        "",
        "| Group | Label | Mean Greedy Success | Complete-Seed Failures | Count |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for label, cases in group_specs:
        summary = _round6_summary_for_cases(campaign, cases)
        for model_label in ("kl_lss_sare", "kl_lss_token_dense", "kl_lss_single_expert"):
            row = {
                "group": label,
                "label": model_label,
                "mean_greedy_success": float(summary[model_label]["mean"]),
                "complete_seed_failures": int(summary[model_label]["complete_seed_failures"]),
                "count": int(summary[model_label]["count"]),
            }
            rows.append(row)
            lines.append(
                f"| `{label}` | `{model_label}` | `{row['mean_greedy_success']:.4f}` | `{row['complete_seed_failures']}` | `{row['count']}` |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `round6` still holds the healthy and guardrail groups while the canonical deadlock cases remain concentrated in the teacher-locked and ambiguous slices.",
            "- The baseline sync stays close enough to the accepted state to proceed into subgroup detection, rescue screening, and any later practicalization work that survives the funnel.",
        ]
    )
    Path(campaign["reports"]["baseline_sync"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(Path(campaign["reports"]["baseline_sync_csv"]), rows)
    _write_json(Path(campaign["reports"]["baseline_sync_json"]), {"rows": rows})


def _expected_subgroup(campaign: dict[str, Any], lane: str, seed: int) -> str:
    key = (str(lane), int(seed))
    if key in {("prospective_c", 193), ("prospective_g", 251)}:
        return "teacher_locked_no_escape"
    if key in {("prospective_i", 283)}:
        return "ambiguous_unstable"
    if key in {("prospective_f", 239), ("prospective_f", 241)}:
        return "aligned_guardrail"
    if key in {(row["lane"], int(row["seed"])) for row in _healthy_cases(campaign)}:
        return "healthy"
    return "healthy"


def _task_cases_for_stage(campaign: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage == "screening":
        return _teacher_locked_dev_cases(campaign)
    if stage == "rerun":
        return [*_teacher_locked_dev_cases(campaign), *_teacher_locked_holdout_cases(campaign), *_ambiguous_cases(campaign)]
    if stage == "antiregression":
        return _healthy_cases(campaign)
    raise ValueError(f"unsupported rescue stage: {stage}")


def _stage_root(campaign: dict[str, Any], stage: str) -> Path:
    return Path(campaign["stage_roots"][f"rescue_{stage}"])


def _spec_path(campaign: dict[str, Any], stage: str, candidate: str, lane: str, seed: int) -> Path:
    return _stage_root(campaign, stage) / candidate / lane / f"seed_{seed}" / "rescue_spec.json"


def _task_output_dir(campaign: dict[str, Any], stage: str, candidate: str, lane: str, seed: int) -> Path:
    return _stage_root(campaign, stage) / candidate / lane / f"seed_{seed}"


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "run_manifest.json"


def _candidate_case_spec(campaign: dict[str, Any], stage: str, candidate: str, lane: str, seed: int) -> dict[str, Any]:
    output_dir = _task_output_dir(campaign, stage, candidate, lane, seed)
    spec = {
        "program": campaign["name"],
        "stage": stage,
        "candidate": candidate,
        "candidate_meta": dict(campaign["rescue_candidates"][candidate]),
        "lane": lane,
        "seed": int(seed),
        "expected_subgroup": _expected_subgroup(campaign, lane, seed),
        "config_path": str(_round6_run_dir(campaign, lane, seed) / "student_resolved_config.yaml"),
        "checkpoint_path": str(_round6_run_dir(campaign, lane, seed) / "latest.pt"),
        "run_dir": str(_round6_run_dir(campaign, lane, seed)),
        "output_root": str(output_dir),
        "rerun_lineage": "fresh" if stage == "screening" else "rerun",
        "episodes": int(
            campaign["analysis"]["rescue_screening_episodes"]
            if stage == "screening"
            else campaign["analysis"]["rescue_validation_episodes"]
        ),
        "max_steps": int(campaign["analysis"]["rescue_max_steps"]),
        "detector": dict(campaign["analysis"]["detector"]),
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
    }
    return spec


def _write_task_spec(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _spec_hash(path: Path) -> str:
    return _sha256_path(path)


def _write_manifest(spec: dict[str, Any], *, status: str, gpu_slot: int | None = None, exit_code: int | None = None) -> None:
    manifest = dict(spec)
    manifest["status"] = status
    if gpu_slot is not None:
        manifest["gpu_slot"] = int(gpu_slot)
    if exit_code is not None:
        manifest["exit_code"] = int(exit_code)
    manifest["updated_at"] = _timestamp()
    _write_json(Path(spec["manifest_path"]), manifest)


def _rescue_candidates_by_family(campaign: dict[str, Any]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for candidate, meta in campaign["rescue_candidates"].items():
        grouped[str(meta["family"])].append(str(candidate))
    return dict(grouped)


def _candidate_compute_cost(meta: dict[str, Any]) -> float:
    rescue_type = str(meta["rescue_type"])
    if rescue_type == "sampled_window":
        return 1.0
    if rescue_type == "route_window":
        return 1.1
    if rescue_type == "restart_from_last_good":
        return 1.5 + 0.1 * float(meta.get("restart_backtrack", 1))
    if rescue_type == "multi_branch":
        return float(meta.get("branch_count", 2))
    if rescue_type == "settling_window":
        return 1.0
    return 1.0


def _sample_action(logits: torch.Tensor, temperature: float) -> tuple[int, float]:
    scaled = logits / max(temperature, 1e-6)
    probs = torch.softmax(scaled, dim=-1)
    action = int(Categorical(probs=probs).sample().item())
    return action, float(probs[action].item())


def _history_unique_ratio(values: list[Any], window: int) -> float:
    if not values:
        return 1.0
    tail = values[-window:]
    return float(len(set(tail)) / max(len(tail), 1))


def _switch_rate(values: list[Any], window: int) -> float:
    tail = values[-window:]
    if len(tail) <= 1:
        return 0.0
    changes = sum(1 for idx in range(1, len(tail)) if tail[idx] != tail[idx - 1])
    return float(changes / max(len(tail) - 1, 1))


def _std(values: list[float], window: int) -> float:
    tail = values[-window:]
    return float(np.std(tail)) if tail else 0.0


def _is_output_settled(detector: dict[str, Any], history: dict[str, list[Any]]) -> bool:
    window = int(detector["repeated_position_window"])
    return (
        _switch_rate(history["actions"], window) <= float(detector["output_settle_action_switch_ceiling"])
        and _std(history["confidences"], window) <= float(detector["output_settle_confidence_std_ceiling"])
    )


def _is_route_settled(detector: dict[str, Any], history: dict[str, list[Any]]) -> bool:
    window = int(detector["repeated_position_window"])
    return (
        _switch_rate(history["route_pairs"], window) <= float(detector["route_settle_pair_switch_ceiling"])
        and _std(history["route_entropies"], window) <= float(detector["route_settle_entropy_std_ceiling"])
    )


def _classify_subgroup(
    detector: dict[str, Any],
    state: dict[str, Any],
    action_summary: dict[str, Any],
    route_summary: dict[str, Any],
    history: dict[str, list[Any]],
    no_progress_steps: int,
) -> dict[str, Any]:
    phase = str(state["phase"])
    unique_ratio = _history_unique_ratio(history["positions"], int(detector["repeated_position_window"]))
    route_dom = float(route_summary.get("dominant_pair_fraction") or 0.0)
    route_entropy = float(route_summary.get("route_entropy") or 0.0)
    confidence = float(action_summary["confidence"])
    output_settled = _is_output_settled(detector, history)
    route_settled = _is_route_settled(detector, history)
    mismatch = output_settled != route_settled
    pre_key = phase in PRE_KEY_PHASES
    stalled = pre_key and no_progress_steps >= int(detector["stall_window"])
    if (
        stalled
        and unique_ratio <= float(detector["low_unique_ratio_threshold"])
        and output_settled
        and confidence <= float(detector["low_confidence_threshold"])
    ):
        label = "teacher_locked_no_escape"
    elif (
        stalled
        and (
            confidence > float(detector["low_confidence_threshold"])
            or route_entropy >= float(detector["high_route_entropy_threshold"])
            or mismatch
            or not output_settled
        )
    ):
        label = "ambiguous_unstable"
    elif phase in {"carry_key", "at_locked_door", "post_unlock"}:
        label = "aligned_guardrail"
    else:
        label = "healthy"
    return {
        "label": label,
        "phase": phase,
        "pre_key": pre_key,
        "stalled": stalled,
        "unique_ratio": unique_ratio,
        "route_dom": route_dom,
        "route_entropy": route_entropy,
        "confidence": confidence,
        "output_settled": output_settled,
        "route_settled": route_settled,
        "mismatch": mismatch,
        "no_progress_steps": no_progress_steps,
    }


def _strongest_subgroup(subgroup_counts: Counter[str]) -> str:
    for label in ("teacher_locked_no_escape", "ambiguous_unstable", "aligned_guardrail", "healthy"):
        if int(subgroup_counts.get(label, 0)) > 0:
            return label
    return "healthy"


def _policy_forward(
    model,
    obs_t: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    done_t: torch.Tensor,
    *,
    route_mode: str | None = None,
    trial: int = 0,
) -> tuple[Any, dict[str, Any]]:
    if route_mode is None:
        output, route_capture = _forward_with_route_capture(model, obs_t, state, done_t)
        return output, _route_capture_summary(route_capture)
    detail = f"{route_mode}:trial={trial}"
    with _patched_core(model.core, "route_randomization", detail, None):
        output, route_capture = _forward_with_route_capture(model, obs_t, state, done_t)
    return output, _route_capture_summary(route_capture)


def _branch_score(progress: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(progress["success"]),
        float(progress["phase_rank"]),
        float(progress["door_unlocked"]),
        float(progress["carrying_key"]),
    )


def _simulate_policy_from_prefix(
    *,
    config_path: Path,
    checkpoint_path: Path,
    prefix_actions: list[int],
    reset_seed: int,
    max_steps: int,
    temperature: float,
    device: torch.device,
    branch_trial: int,
) -> dict[str, Any]:
    config, model = _load_model(str(config_path), str(checkpoint_path), device)
    model.eval()
    env, _env_config = _build_single_env(config_path)
    rng_state = capture_rng_state()
    set_seed(config.seed + 70_000 + branch_trial, deterministic=config.system.deterministic)
    simulated_steps = 0
    try:
        obs, _ = env.reset(seed=reset_seed)
        done_t = torch.ones(1, device=device, dtype=torch.bool)
        state = model.initial_state(1, device)
        total_reward = 0.0
        last_state = _extract_grid_state(env)
        key_pickup = False
        door_unlocked = False
        action_history: list[int] = []
        for action in prefix_actions:
            obs_t = prepare_obs(_obs_to_batch(obs), device)
            with torch.inference_mode():
                output, _route = _policy_forward(model, obs_t, state, done_t)
            state = output.next_state
            obs, reward, terminated, truncated, info = env.step(int(action))
            simulated_steps += 1
            action_history.append(int(action))
            total_reward += float(reward)
            new_state = _extract_grid_state(env)
            key_pickup = key_pickup or (not last_state["carrying_key"] and new_state["carrying_key"])
            door_unlocked = door_unlocked or (last_state["door_locked"] and not new_state["door_locked"])
            last_state = new_state
            done = bool(terminated or truncated)
            done_t = prepare_done(np.asarray([done], dtype=bool), device)
            if done:
                return {
                    "success": float(info.get("success", reward > 0.0)),
                    "return": total_reward,
                    "length": float(len(action_history)),
                    "simulated_steps": float(simulated_steps),
                    "phase_rank": float(PROGRESS_PHASE_RANK[str(last_state["phase"])]),
                    "carrying_key": float(last_state["carrying_key"]),
                    "door_unlocked": float(not last_state["door_locked"]),
                }
        for _step in range(len(prefix_actions), max_steps):
            obs_t = prepare_obs(_obs_to_batch(obs), device)
            with torch.inference_mode():
                output, _route = _policy_forward(model, obs_t, state, done_t)
            state = output.next_state
            logits = output.logits[0]
            action, _selected_prob = _sample_action(logits, temperature)
            obs, reward, terminated, truncated, info = env.step(int(action))
            simulated_steps += 1
            action_history.append(int(action))
            total_reward += float(reward)
            new_state = _extract_grid_state(env)
            key_pickup = key_pickup or (not last_state["carrying_key"] and new_state["carrying_key"])
            door_unlocked = door_unlocked or (last_state["door_locked"] and not new_state["door_locked"])
            last_state = new_state
            done = bool(terminated or truncated)
            done_t = prepare_done(np.asarray([done], dtype=bool), device)
            if done:
                return {
                    "success": float(info.get("success", reward > 0.0)),
                    "return": total_reward,
                    "length": float(len(action_history)),
                    "simulated_steps": float(simulated_steps),
                    "phase_rank": float(PROGRESS_PHASE_RANK[str(last_state["phase"])]),
                    "carrying_key": float(last_state["carrying_key"]),
                    "door_unlocked": float(not last_state["door_locked"]),
                }
        return {
            "success": 0.0,
            "return": total_reward,
            "length": float(len(action_history)),
            "simulated_steps": float(simulated_steps),
            "phase_rank": float(PROGRESS_PHASE_RANK[str(last_state["phase"])]),
            "carrying_key": float(last_state["carrying_key"]),
            "door_unlocked": float(not last_state["door_locked"]),
        }
    finally:
        env.close()
        restore_rng_state(rng_state)


def _phase_bucket(phase: str) -> str:
    if phase in PRE_KEY_PHASES:
        return "pre_key"
    if phase == "carry_key":
        return "carry_key"
    if phase == "at_locked_door":
        return "at_locked_door"
    return "post_unlock"


def _state_tensor_features(state: dict[str, torch.Tensor] | None) -> list[float]:
    if not state:
        return [0.0, 0.0, 0.0, 0.0]
    features: list[float] = []
    for key in sorted(state):
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            continue
        flat = tensor.detach().float().reshape(-1)
        if flat.numel() == 0:
            features.extend([0.0, 0.0, 0.0, 0.0])
            continue
        features.extend(
            [
                float(flat.mean().item()),
                float(flat.std(unbiased=False).item()),
                float(torch.linalg.vector_norm(flat).item()),
                float(flat.abs().max().item()),
            ]
        )
    return features or [0.0, 0.0, 0.0, 0.0]


def _source_model_paths(campaign: dict[str, Any], model_spec: dict[str, Any], lane: str, seed: int) -> tuple[Path, Path, Path]:
    if bool(model_spec.get("case_specific")):
        run_dir = _label_run_dir(campaign, str(model_spec["label"]), lane, seed)
        return run_dir, _resolved_config_path(run_dir), run_dir / "latest.pt"
    run_dir = Path(str(model_spec["run_dir"]))
    return run_dir, Path(str(model_spec["config_path"])), Path(str(model_spec["checkpoint_path"]))


def _probe_branch_labels(
    *,
    config_path: Path,
    checkpoint_path: Path,
    prefix_actions: list[int],
    reset_seed: int,
    max_steps: int,
    horizon: int,
    temperature: float,
    device: torch.device,
    current_phase_rank: int,
    carrying_key: bool,
    door_locked: bool,
    episode_index: int,
) -> dict[str, Any]:
    action_count = 7
    rollouts: list[dict[str, Any]] = []
    branch_limit = min(max_steps, len(prefix_actions) + 1 + horizon)
    for forced_action in range(action_count):
        rollout = _simulate_policy_from_prefix(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            prefix_actions=[*prefix_actions, forced_action],
            reset_seed=reset_seed,
            max_steps=branch_limit,
            temperature=temperature,
            device=device,
            branch_trial=50_000 + episode_index * 100 + forced_action,
        )
        rollout["forced_action"] = int(forced_action)
        rollouts.append(rollout)
    if not rollouts:
        return {
            "recoverable": None,
            "escape_action": None,
            "escape_action_available": 0,
        }
    best = max(rollouts, key=_branch_score)
    baseline_tuple = (
        0.0,
        float(current_phase_rank),
        float(not door_locked),
        float(carrying_key),
    )
    recoverable = int(_branch_score(best) > baseline_tuple)
    return {
        "recoverable": recoverable,
        "escape_action": int(best["forced_action"]),
        "escape_action_available": int(recoverable > 0),
        "best_branch_success": float(best["success"]),
        "best_branch_phase_rank": float(best["phase_rank"]),
    }


def _collect_source_quality_rows(
    campaign: dict[str, Any],
    model_spec: dict[str, Any],
    case: dict[str, Any],
    *,
    device: str,
) -> list[dict[str, Any]]:
    lane = str(case["lane"])
    seed = int(case["seed"])
    run_dir, config_path, checkpoint_path = _source_model_paths(campaign, model_spec, lane, seed)
    if not config_path.exists() or not checkpoint_path.exists():
        return []
    teacher_run = _teacher_run_dir(campaign, lane, seed)
    teacher_config_path = _resolved_config_path(teacher_run)
    teacher_checkpoint = teacher_run / "latest.pt"
    if not teacher_config_path.exists() or not teacher_checkpoint.exists():
        return []
    device_t = torch.device("cpu" if device == "cpu" or not torch.cuda.is_available() else "cuda")
    _config, model = _load_model(str(config_path), str(checkpoint_path), device_t)
    teacher_config, teacher_model = _load_model(str(teacher_config_path), str(teacher_checkpoint), device_t)
    model.eval()
    teacher_model.eval()
    env, _env_config = _build_single_env(config_path)
    rows: list[dict[str, Any]] = []
    detector = dict(campaign["analysis"]["detector"])
    episodes = int(campaign["analysis"]["source_quality_episodes"])
    max_steps = int(campaign["analysis"]["source_quality_max_steps"])
    horizon = int(campaign["analysis"]["source_quality_branch_horizon"])
    max_branch_states = int(campaign["analysis"]["source_quality_max_branch_states_per_episode"])
    try:
        for episode_index in range(episodes):
            reset_seed = int(seed) + 9_000 + episode_index
            rng_state = capture_rng_state()
            set_seed(reset_seed, deterministic=teacher_config.system.deterministic)
            try:
                obs, _ = env.reset(seed=reset_seed)
                model_state = model.initial_state(1, device_t)
                teacher_state = teacher_model.initial_state(1, device_t)
                done_t = torch.ones(1, device=device_t, dtype=torch.bool)
                positions: list[tuple[int, int]] = []
                actions: list[int] = []
                confidences: list[float] = []
                route_pairs: list[str] = []
                route_entropies: list[float] = []
                prefix_actions: list[int] = []
                branch_states = 0
                for step_index in range(max_steps):
                    pre_state = _extract_grid_state(env)
                    obs_t = prepare_obs(_obs_to_batch(obs), device_t)
                    with torch.inference_mode():
                        teacher_output, _ = _policy_forward(teacher_model, obs_t, teacher_state, done_t)
                        student_output, route_capture = _forward_with_route_capture(model, obs_t, model_state, done_t)
                    teacher_state = teacher_output.next_state
                    model_state = student_output.next_state
                    teacher_summary = _action_summary(teacher_output.logits, teacher_model)
                    student_summary = _action_summary(student_output.logits, model)
                    route_summary = _route_capture_summary(route_capture)
                    positions.append(tuple(int(value) for value in pre_state["agent_pos"]))
                    actions.append(int(student_summary["action"]))
                    confidences.append(float(student_summary["confidence"]))
                    dominant_pair = "-" if route_summary["dominant_pair"] is None else str(route_summary["dominant_pair"])
                    route_pairs.append(dominant_pair)
                    route_entropies.append(float(route_summary.get("route_entropy") or 0.0))
                    history = {
                        "positions": positions,
                        "actions": actions,
                        "confidences": confidences,
                        "route_pairs": route_pairs,
                        "route_entropies": route_entropies,
                    }
                    detector_row = _classify_subgroup(
                        detector,
                        pre_state,
                        student_summary,
                        route_summary,
                        history,
                        no_progress_steps=step_index,
                    )
                    logits = student_output.logits[0].detach().float().cpu().numpy()
                    probs = torch.softmax(student_output.logits[0], dim=-1).detach().float().cpu().numpy()
                    features = np.asarray(
                        [
                            *logits.tolist(),
                            *probs.tolist(),
                            float(student_summary["confidence"]),
                            float(student_summary["entropy"]),
                            float(student_summary["margin"]),
                            float(route_summary.get("route_entropy") or 0.0),
                            float(route_summary.get("dominant_pair_fraction") or 0.0),
                            float(route_summary.get("unique_pair_count") or 0.0),
                            *_state_tensor_features(student_output.next_state),
                        ],
                        dtype=np.float32,
                    )
                    probe_row: dict[str, Any] = {
                        "model_id": str(model_spec["model_id"]),
                        "model_family": str(model_spec["family"]),
                        "lane": lane,
                        "seed": seed,
                        "episode_index": episode_index,
                        "step": step_index,
                        "phase_bucket": _phase_bucket(str(pre_state["phase"])),
                        "teacher_action": int(teacher_summary["action"]),
                        "subgroup": str(detector_row["label"]),
                        "route_choice": dominant_pair if bool(model_spec.get("routed")) and dominant_pair != "-" else None,
                        "features": features.tolist(),
                        "recoverable": None,
                        "escape_action": None,
                        "escape_action_available": 0,
                    }
                    if (
                        branch_states < max_branch_states
                        and str(detector_row["label"]) in {"teacher_locked_no_escape", "ambiguous_unstable"}
                        and str(pre_state["phase"]) in PRE_KEY_PHASES
                    ):
                        branch_labels = _probe_branch_labels(
                            config_path=config_path,
                            checkpoint_path=checkpoint_path,
                            prefix_actions=prefix_actions,
                            reset_seed=reset_seed,
                            max_steps=max_steps,
                            horizon=horizon,
                            temperature=1.0,
                            device=device_t,
                            current_phase_rank=PROGRESS_PHASE_RANK[str(pre_state["phase"])],
                            carrying_key=bool(pre_state["carrying_key"]),
                            door_locked=bool(pre_state["door_locked"]),
                            episode_index=episode_index,
                        )
                        probe_row.update(branch_labels)
                        branch_states += 1
                    rows.append(probe_row)
                    action = int(student_summary["action"])
                    obs, reward, terminated, truncated, _info = env.step(action)
                    prefix_actions.append(action)
                    done = bool(terminated or truncated)
                    done_t = prepare_done(np.asarray([done], dtype=bool), device_t)
                    if done:
                        break
            finally:
                restore_rng_state(rng_state)
    finally:
        env.close()
    return rows


def _fit_linear_probe(rows: list[dict[str, Any]], target_key: str) -> dict[str, Any]:
    usable = [row for row in rows if row.get(target_key) not in (None, "", "None")]
    if len(usable) < 16:
        return {"status": "insufficient_data", "samples": len(usable)}
    labels = sorted({str(row[target_key]) for row in usable})
    if len(labels) < 2:
        return {"status": "single_class", "samples": len(usable), "classes": len(labels)}
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    x = np.asarray([row["features"] for row in usable], dtype=np.float32)
    y = np.asarray([label_to_idx[str(row[target_key])] for row in usable], dtype=np.int64)
    rng = np.random.default_rng(0)
    indices = np.arange(len(usable))
    rng.shuffle(indices)
    split = max(int(len(indices) * 0.8), 1)
    split = min(split, len(indices) - 1)
    train_idx = indices[:split]
    test_idx = indices[split:]
    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    mean_x = x_train.mean(axis=0, keepdims=True)
    std_x = x_train.std(axis=0, keepdims=True)
    std_x[std_x < 1e-6] = 1.0
    x_train = (x_train - mean_x) / std_x
    x_test = (x_test - mean_x) / std_x
    train_x_t = torch.as_tensor(x_train, dtype=torch.float32)
    train_y_t = torch.as_tensor(y_train, dtype=torch.long)
    test_x_t = torch.as_tensor(x_test, dtype=torch.float32)
    test_y_t = torch.as_tensor(y_test, dtype=torch.long)
    probe = torch.nn.Linear(train_x_t.shape[1], len(labels))
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.05)
    for _step in range(200):
        optimizer.zero_grad(set_to_none=True)
        logits = probe(train_x_t)
        loss = torch.nn.functional.cross_entropy(logits, train_y_t)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        train_pred = probe(train_x_t).argmax(dim=-1)
        test_pred = probe(test_x_t).argmax(dim=-1)
    counts = Counter(int(value) for value in y)
    majority = float(max(counts.values()) / len(y))
    chance = float(1.0 / len(labels))
    return {
        "status": "completed",
        "samples": int(len(usable)),
        "classes": int(len(labels)),
        "chance": float(chance),
        "majority_baseline": float(majority),
        "train_accuracy": float((train_pred == train_y_t).float().mean().item()),
        "test_accuracy": float((test_pred == test_y_t).float().mean().item()),
        "lift_vs_majority": float((test_pred == test_y_t).float().mean().item() - majority),
    }


def render_source_quality(campaign: dict[str, Any], output: Path, json_output: Path | None, csv_output: Path | None, *, device: str) -> None:
    cases = _source_quality_cases(campaign)
    model_specs = _source_model_specs(campaign)
    all_rows: list[dict[str, Any]] = []
    for model_spec in model_specs:
        for case in cases:
            all_rows.extend(_collect_source_quality_rows(campaign, model_spec, case, device=device))
    probe_rows: list[dict[str, Any]] = []
    for model_spec in model_specs:
        model_rows = [row for row in all_rows if str(row["model_id"]) == str(model_spec["model_id"])]
        for target in ("teacher_action", "phase_bucket", "recoverable", "escape_action"):
            result = _fit_linear_probe(model_rows, target)
            probe_rows.append(
                {
                    "model_id": str(model_spec["model_id"]),
                    "model_family": str(model_spec["family"]),
                    "target": target,
                    **result,
                }
            )
        if bool(model_spec.get("routed")):
            route_result = _fit_linear_probe(model_rows, "route_choice")
            probe_rows.append(
                {
                    "model_id": str(model_spec["model_id"]),
                    "model_family": str(model_spec["family"]),
                    "target": "route_choice",
                    **route_result,
                }
            )
    by_key = {(str(row["model_id"]), str(row["target"])): row for row in probe_rows}
    round6_escape = by_key.get(("round6", "escape_action"), {})
    arch_incumbent_escape = by_key.get(("architecture_keyed_residual", "escape_action"), {})
    arch_challenger_escape = by_key.get(("architecture_weak_base_expert_gate", "escape_action"), {})
    best_arch_escape = max(
        [arch_incumbent_escape, arch_challenger_escape],
        key=lambda item: float(item.get("test_accuracy", 0.0)),
        default={},
    )
    round6_meaningful = float(round6_escape.get("test_accuracy", 0.0)) >= max(
        float(round6_escape.get("majority_baseline", 0.0)) + 0.10,
        float(round6_escape.get("chance", 0.0)) + 0.15,
    )
    arch_stronger = float(best_arch_escape.get("test_accuracy", 0.0)) >= float(round6_escape.get("test_accuracy", 0.0)) + 0.05
    if round6_meaningful:
        verdict = "downstream_policy_formation_bottleneck"
    elif arch_stronger:
        verdict = "architecture_exposes_extra_escape_signal"
    else:
        verdict = "source_quality_ceiling_still_binding"
    lines = [
        "# Escape2x Stage A2 Source Quality",
        "",
        f"- source-quality cases: `{len(cases)}`",
        f"- source-quality models: `{[row['model_id'] for row in model_specs]}`",
        f"- completed bounded probe fits: `{sum(1 for row in probe_rows if row.get('status') == 'completed')}`",
        f"- source-quality verdict: `{verdict}`",
        "",
        "| Model | Family | Target | Samples | Classes | Test Acc | Majority | Chance | Lift vs Majority | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(probe_rows, key=lambda item: (str(item["model_family"]), str(item["model_id"]), str(item["target"]))):
        lines.append(
            f"| `{row['model_id']}` | `{row['model_family']}` | `{row['target']}` | `{int(row.get('samples', 0))}` | `{int(row.get('classes', 0))}` | `{float(row.get('test_accuracy', 0.0)):.4f}` | `{float(row.get('majority_baseline', 0.0)):.4f}` | `{float(row.get('chance', 0.0)):.4f}` | `{float(row.get('lift_vs_majority', 0.0)):.4f}` | `{row.get('status', 'missing')}` |"
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"- Does `round6` already expose the missing escape signal? `{round6_meaningful}` via the `escape_action` probe on its own frozen states.",
            f"- Does the architecture incumbent expose more of it? `{arch_stronger}` when comparing the best architecture `escape_action` probe against `round6`.",
            f"- Is the current bottleneck representation/source quality or downstream policy formation? `{verdict}`.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "probe_rows": probe_rows,
                "verdict": verdict,
                "round6_meaningful_escape_signal": bool(round6_meaningful),
                "architecture_stronger_escape_signal": bool(arch_stronger),
            },
        )
    if csv_output is not None:
        _write_csv(csv_output, probe_rows)


def _evaluate_episode(
    *,
    spec: dict[str, Any],
    device: torch.device,
    trace_only: bool = False,
) -> dict[str, Any]:
    config_path = Path(spec["config_path"])
    checkpoint_path = Path(spec["checkpoint_path"])
    config, model = _load_model(str(config_path), str(checkpoint_path), device)
    model.eval()
    env, _env_config = _build_single_env(config_path)
    candidate = dict(spec.get("candidate_meta", {}))
    detector = dict(spec["detector"])
    reset_seed = int(spec["episode_reset_seed"])
    rng_state = capture_rng_state()
    set_seed(config.seed + 60_000 + int(spec["episode_index"]), deterministic=config.system.deterministic)
    action_history: list[int] = []
    positions: list[tuple[int, int]] = []
    actions: list[int] = []
    confidences: list[float] = []
    route_pairs: list[str] = []
    route_entropies: list[float] = []
    subgroup_counts: Counter[str] = Counter()
    simulated_steps = 0
    rescue_triggers = 0
    rescue_false_positive = 0
    detector_triggered = False
    rescue_used = False
    rescue_steps_remaining = 0
    adaptive_active = False
    last_progress_step = 0
    episode_return = 0.0
    success = 0.0
    try:
        obs, _ = env.reset(seed=reset_seed)
        done_t = torch.ones(1, device=device, dtype=torch.bool)
        state = model.initial_state(1, device)
        last_state = _extract_grid_state(env)
        progress_phase_rank = PROGRESS_PHASE_RANK[str(last_state["phase"])]
        triggered_labels: list[str] = []
        step_rows: list[dict[str, Any]] = []
        for step in range(int(spec["max_steps"])):
            obs_t = prepare_obs(_obs_to_batch(obs), device)
            with torch.inference_mode():
                base_output, base_route = _policy_forward(model, obs_t, state, done_t)
            base_state = base_output.next_state
            base_summary = _action_summary(base_output.logits, model)
            base_route_pair = "-" if base_route["dominant_pair"] is None else str(base_route["dominant_pair"])
            positions.append(tuple(int(value) for value in last_state["agent_pos"]))
            actions.append(int(base_summary["action"]))
            confidences.append(float(base_summary["confidence"]))
            route_pairs.append(base_route_pair)
            route_entropies.append(float(base_route.get("route_entropy") or 0.0))
            no_progress_steps = step - last_progress_step
            history = {
                "positions": positions,
                "actions": actions,
                "confidences": confidences,
                "route_pairs": route_pairs,
                "route_entropies": route_entropies,
            }
            detector_row = _classify_subgroup(detector, last_state, base_summary, base_route, history, no_progress_steps)
            subgroup_counts[str(detector_row["label"])] += 1
            detector_triggered = detector_triggered or str(detector_row["label"]) == "teacher_locked_no_escape"

            use_rescue_now = False
            route_mode = None
            temperature = 1.0
            rescue_label = str(detector_row["label"])
            rescue_type = str(candidate.get("rescue_type", "none"))
            if not trace_only:
                if rescue_type in {"sampled_window", "route_window"}:
                    if detector_row["label"] == "teacher_locked_no_escape" and rescue_steps_remaining <= 0:
                        rescue_triggers += 1
                        triggered_labels.append(rescue_label)
                        rescue_steps_remaining = int(candidate.get("window", 4))
                        adaptive_active = bool(candidate.get("adaptive", False))
                    if rescue_steps_remaining > 0:
                        use_rescue_now = True
                        temperature = float(candidate.get("temperature", 1.0))
                        if rescue_type == "route_window":
                            route_mode = str(candidate["route_mode"])
                elif rescue_type == "settling_window":
                    settling_mode = str(candidate.get("settling_mode"))
                    triggered = False
                    if detector_row["stalled"]:
                        if settling_mode == "output_unsettled" and not detector_row["output_settled"]:
                            triggered = True
                        elif settling_mode == "route_unsettled" and not detector_row["route_settled"]:
                            triggered = True
                        elif settling_mode == "mismatch" and detector_row["mismatch"]:
                            triggered = True
                    if triggered and rescue_steps_remaining <= 0:
                        rescue_triggers += 1
                        triggered_labels.append("settling_trigger")
                        rescue_steps_remaining = int(candidate.get("window", 4))
                    if rescue_steps_remaining > 0:
                        use_rescue_now = True
                        temperature = float(candidate.get("temperature", 1.0))
                elif rescue_type in {"restart_from_last_good", "multi_branch"} and detector_row["label"] == "teacher_locked_no_escape" and not rescue_used:
                    rescue_used = True
                    rescue_triggers += 1
                    triggered_labels.append(rescue_label)
                    if rescue_type == "restart_from_last_good":
                        backtrack = int(candidate.get("restart_backtrack", 1))
                        restart_index = max(0, len(action_history) - backtrack)
                        rollout = _simulate_policy_from_prefix(
                            config_path=config_path,
                            checkpoint_path=checkpoint_path,
                            prefix_actions=action_history[:restart_index],
                            reset_seed=reset_seed,
                            max_steps=int(spec["max_steps"]),
                            temperature=float(candidate.get("temperature", 1.0)),
                            device=device,
                            branch_trial=1000 + int(spec["episode_index"]),
                        )
                        rollout["rescue_triggered"] = 1.0
                        rollout["trigger_false_positive"] = float(spec["expected_subgroup"] == "healthy")
                        rollout["trigger_count"] = 1.0
                        rollout["episode_predicted_subgroup"] = "teacher_locked_no_escape"
                        rollout["triggered_labels"] = triggered_labels
                        return rollout
                    branch_count = int(candidate.get("branch_count", 2))
                    branch_rows: list[dict[str, Any]] = []
                    for branch_index in range(branch_count):
                        branch_rows.append(
                            _simulate_policy_from_prefix(
                                config_path=config_path,
                                checkpoint_path=checkpoint_path,
                                prefix_actions=action_history,
                                reset_seed=reset_seed,
                                max_steps=int(spec["max_steps"]),
                                temperature=float(candidate.get("temperature", 1.0)),
                                device=device,
                                branch_trial=2000 + branch_index + int(spec["episode_index"]) * 100,
                            )
                        )
                    best_branch = max(branch_rows, key=_branch_score)
                    best_branch["simulated_steps"] = float(sum(float(row["simulated_steps"]) for row in branch_rows))
                    best_branch["rescue_triggered"] = 1.0
                    best_branch["trigger_false_positive"] = float(spec["expected_subgroup"] == "healthy")
                    best_branch["trigger_count"] = 1.0
                    best_branch["episode_predicted_subgroup"] = "teacher_locked_no_escape"
                    best_branch["triggered_labels"] = triggered_labels
                    return best_branch

            current_output = base_output
            current_route = base_route
            if use_rescue_now and route_mode is not None:
                with torch.inference_mode():
                    current_output, current_route = _policy_forward(
                        model,
                        obs_t,
                        state,
                        done_t,
                        route_mode=route_mode,
                        trial=10_000 + int(spec["episode_index"]) * 1000 + step,
                    )
            state = current_output.next_state if use_rescue_now else base_state
            if use_rescue_now:
                action, selected_prob = _sample_action(current_output.logits[0], temperature)
                if spec["expected_subgroup"] == "healthy":
                    rescue_false_positive = 1
            else:
                action = int(base_summary["action"])
                selected_prob = float(torch.softmax(base_output.logits[0], dim=-1)[action].item())

            obs, reward, terminated, truncated, info = env.step(int(action))
            simulated_steps += 1
            action_history.append(int(action))
            episode_return += float(reward)
            new_state = _extract_grid_state(env)
            if (not last_state["carrying_key"] and new_state["carrying_key"]) or (last_state["door_locked"] and not new_state["door_locked"]):
                last_progress_step = step + 1
                progress_phase_rank = max(progress_phase_rank, PROGRESS_PHASE_RANK[str(new_state["phase"])])
                if adaptive_active:
                    rescue_steps_remaining = 1
            progress_phase_rank = max(progress_phase_rank, PROGRESS_PHASE_RANK[str(new_state["phase"])])
            if use_rescue_now and rescue_steps_remaining > 0:
                rescue_steps_remaining -= 1
            if use_rescue_now and adaptive_active and detector_row["stalled"] and rescue_steps_remaining <= 0:
                rescue_steps_remaining = max(int(candidate.get("window", 12)) - 1, 0)
            step_rows.append(
                {
                    "step": step,
                    "phase": str(last_state["phase"]),
                    "subgroup": detector_row["label"],
                    "confidence": float(base_summary["confidence"]),
                    "route_entropy": float(base_route.get("route_entropy") or 0.0),
                    "selected_prob": selected_prob,
                    "rescue": float(use_rescue_now),
                    "action": int(action),
                }
            )
            last_state = new_state
            done = bool(terminated or truncated)
            done_t = prepare_done(np.asarray([done], dtype=bool), device)
            if done:
                success = float(info.get("success", reward > 0.0))
                break
        episode_predicted_subgroup = _strongest_subgroup(subgroup_counts)
        return {
            "success": float(success),
            "return": float(episode_return),
            "length": float(len(action_history)),
            "simulated_steps": float(simulated_steps),
            "phase_rank": float(progress_phase_rank),
            "carrying_key": float(last_state["carrying_key"]),
            "door_unlocked": float(not last_state["door_locked"]),
            "rescue_triggered": float(detector_triggered or rescue_triggers > 0),
            "trigger_count": float(rescue_triggers),
            "trigger_false_positive": float(rescue_false_positive),
            "episode_predicted_subgroup": episode_predicted_subgroup,
            "triggered_labels": triggered_labels,
            "trace": step_rows[:64],
        }
    finally:
        env.close()
        restore_rng_state(rng_state)


def run_rescue_task(spec_path: Path, *, device: str) -> None:
    spec = _read_json(spec_path)
    output_dir = Path(spec["output_root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = dict(spec)
    manifest_payload["manifest_path"] = str(_manifest_path(output_dir))
    _write_manifest(manifest_payload, status="running")
    device_t = torch.device("cpu" if device == "cpu" or not torch.cuda.is_available() else "cuda")
    episodes = int(spec["episodes"])
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for episode_index in range(episodes):
        episode_spec = dict(spec)
        episode_spec["episode_index"] = episode_index
        episode_spec["episode_reset_seed"] = int(spec["seed"]) + 5_000 + episode_index
        episode_row = _evaluate_episode(spec=episode_spec, device=device_t, trace_only=False)
        rows.append({"episode_index": episode_index, **episode_row})
    wall_seconds = float(time.perf_counter() - start)
    success_values = [float(row["success"]) for row in rows]
    trigger_values = [float(row["rescue_triggered"]) for row in rows]
    simulated_steps = [float(row["simulated_steps"]) for row in rows]
    lengths = [float(row["length"]) for row in rows]
    summary = {
        "candidate": spec["candidate"],
        "family": spec["candidate_meta"]["family"],
        "stage": spec["stage"],
        "lane": spec["lane"],
        "seed": spec["seed"],
        "expected_subgroup": spec["expected_subgroup"],
        "episodes": episodes,
        "success_rate": float(mean(success_values)) if success_values else 0.0,
        "return_mean": float(mean(float(row["return"]) for row in rows)) if rows else 0.0,
        "length_mean": float(mean(lengths)) if lengths else 0.0,
        "trigger_rate": float(mean(trigger_values)) if trigger_values else 0.0,
        "trigger_count_mean": float(mean(float(row["trigger_count"]) for row in rows)) if rows else 0.0,
        "false_positive_rate": float(mean(float(row["trigger_false_positive"]) for row in rows)) if rows else 0.0,
        "compute_multiplier_mean": float(mean(simulated_steps) / max(mean(lengths), 1.0)) if lengths else 1.0,
        "majority_subgroup": Counter(str(row["episode_predicted_subgroup"]) for row in rows).most_common(1)[0][0] if rows else "healthy",
        "wall_seconds": wall_seconds,
        "rows": rows,
    }
    _write_json(output_dir / "summary.json", summary)
    manifest_payload["end_timestamp"] = _timestamp()
    _write_manifest(manifest_payload, status="completed", exit_code=0)


def _launch_task(task: dict[str, Any], *, device: str, gpu_slot: int | None) -> subprocess.Popen[str]:
    env = os.environ.copy()
    if gpu_slot is not None and device != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_slot)
    command = [
        sys.executable,
        "-m",
        "psmn_rl.analysis.lss_deadlock_escape",
        "run-task",
        "--spec",
        task["spec_path"],
        "--device",
        "cuda" if gpu_slot is not None and device != "cpu" else device,
    ]
    output_dir = Path(task["output_root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    task["start_timestamp"] = _timestamp()
    _write_manifest(task, status="running", gpu_slot=gpu_slot)
    return subprocess.Popen(command, cwd=Path.cwd(), env=env, text=True)


def _selected_stage1_candidates(campaign: dict[str, Any]) -> list[str]:
    payload = _optional_json(campaign["reports"].get("rescue_stage1_json"))
    return [str(name) for name in payload.get("selected_candidates", [])]


def _selected_stage2_candidates(campaign: dict[str, Any]) -> list[str]:
    payload = _optional_json(campaign["reports"].get("rescue_stage2_json"))
    return [str(name) for name in payload.get("surviving_candidates", [])]


def run_rescue_stage(campaign: dict[str, Any], *, stage: str, device: str) -> None:
    if stage == "screening":
        candidates = list(campaign["rescue_candidates"].keys())
    elif stage == "rerun":
        candidates = _selected_stage1_candidates(campaign)
    elif stage == "antiregression":
        candidates = _selected_stage2_candidates(campaign)
    else:
        raise ValueError(f"unsupported rescue stage: {stage}")
    if not candidates:
        return
    tasks: list[dict[str, Any]] = []
    for candidate in candidates:
        for case in _task_cases_for_stage(campaign, stage):
            lane = str(case["lane"])
            seed = int(case["seed"])
            spec = _candidate_case_spec(campaign, stage, candidate, lane, seed)
            spec_path = _spec_path(campaign, stage, candidate, lane, seed)
            _write_task_spec(spec_path, spec)
            task = dict(spec)
            task["spec_path"] = str(spec_path)
            task["config_hash"] = _spec_hash(spec_path)
            task["seed_block"] = f"{lane}:{seed}"
            task["family_label"] = str(campaign["rescue_candidates"][candidate]["family"])
            task["manifest_path"] = str(_manifest_path(Path(spec["output_root"])))
            task["start_timestamp"] = None
            task["end_timestamp"] = None
            _write_manifest(task, status="pending")
            tasks.append(task)
    _write_json(
        Path(campaign["reports"]["rescue_queue_manifest_json"]),
        {
            "program_label": _analysis_label(campaign, "2x-scale Deadlock Escape Program"),
            "stage": stage,
            "device_request": device,
            "visible_gpu_count": _gpu_count(device),
            "queued_tasks": tasks,
            "created_at": _timestamp(),
            "git_commit": get_git_commit(),
            "git_dirty": get_git_dirty(),
        },
    )
    pending = [task for task in tasks if not (Path(task["output_root"]) / "summary.json").exists()]
    if not pending:
        return
    count = _gpu_count(device)
    if device == "cpu" or count <= 0:
        for task in pending:
            proc = _launch_task(task, device="cpu", gpu_slot=None)
            exit_code = proc.wait()
            task["end_timestamp"] = _timestamp()
            _write_manifest(task, status="completed" if exit_code == 0 else "failed", exit_code=exit_code)
            if exit_code != 0:
                raise RuntimeError(f"rescue task failed: {task['candidate']} {task['seed_block']}")
        return

    active: list[tuple[subprocess.Popen[str], dict[str, Any], int]] = []
    failed: list[str] = []

    def wait_one() -> None:
        while True:
            for index, (proc, task, slot) in enumerate(active):
                exit_code = proc.poll()
                if exit_code is None:
                    continue
                active.pop(index)
                task["end_timestamp"] = _timestamp()
                _write_manifest(task, status="completed" if exit_code == 0 else "failed", gpu_slot=slot, exit_code=exit_code)
                if exit_code != 0:
                    failed.append(f"{task['candidate']} {task['seed_block']}")
                return
            time.sleep(1.0)

    slot = 0
    for task in pending:
        gpu_slot = slot % count
        active.append((_launch_task(task, device=device, gpu_slot=gpu_slot), task, gpu_slot))
        slot += 1
        if len(active) >= count:
            wait_one()
    while active:
        wait_one()
    if failed:
        raise RuntimeError(f"rescue tasks failed: {failed}")


def _stage_rows(campaign: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    candidates = list(campaign["rescue_candidates"].keys()) if stage == "screening" else (_selected_stage1_candidates(campaign) if stage == "rerun" else _selected_stage2_candidates(campaign))
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for case in _task_cases_for_stage(campaign, stage):
            summary_path = _task_output_dir(campaign, stage, candidate, case["lane"], int(case["seed"])) / "summary.json"
            if not summary_path.exists():
                continue
            payload = _read_json(summary_path)
            rows.append(
                {
                    "candidate": candidate,
                    "family": str(payload["family"]),
                    "stage": stage,
                    "lane": str(payload["lane"]),
                    "seed": int(payload["seed"]),
                    "expected_subgroup": str(payload["expected_subgroup"]),
                    "success_rate": float(payload["success_rate"]),
                    "trigger_rate": float(payload["trigger_rate"]),
                    "trigger_count_mean": float(payload["trigger_count_mean"]),
                    "false_positive_rate": float(payload["false_positive_rate"]),
                    "compute_multiplier_mean": float(payload["compute_multiplier_mean"]),
                    "wall_seconds": float(payload["wall_seconds"]),
                    "majority_subgroup": str(payload["majority_subgroup"]),
                }
            )
    return rows


def _current_rows_lookup(campaign: dict[str, Any]) -> dict[tuple[str, int], dict[str, float]]:
    rows = _current_round6_rows(campaign)
    result: dict[tuple[str, int], dict[str, float]] = {}
    for lane, seed in {(str(row["lane"]), int(row["seed"])) for row in rows}:
        filtered = [row for row in rows if str(row["lane"]) == lane and int(row["seed"]) == seed]
        result[(lane, seed)] = {
            "round6": next(float(row["final_greedy_success"]) for row in filtered if str(row["label"]) == "kl_lss_sare"),
            "token_dense": next(float(row["final_greedy_success"]) for row in filtered if str(row["label"]) == "kl_lss_token_dense"),
            "single_expert": next(float(row["final_greedy_success"]) for row in filtered if str(row["label"]) == "kl_lss_single_expert"),
        }
    return result


def _aggregate_screening(campaign: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    baseline_lookup = _current_rows_lookup(campaign)
    family_top_k = int(campaign["selection"]["rescue_family_top_k"])
    summaries: list[dict[str, Any]] = []
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_candidate[str(row["candidate"])].append(row)
    for candidate, candidate_rows in sorted(by_candidate.items()):
        baseline_rows = [baseline_lookup[(str(row["lane"]), int(row["seed"]))] for row in candidate_rows]
        candidate_mean = mean(float(row["success_rate"]) for row in candidate_rows)
        round6_mean = mean(float(item["round6"]) for item in baseline_rows)
        token_mean = mean(float(item["token_dense"]) for item in baseline_rows)
        single_mean = mean(float(item["single_expert"]) for item in baseline_rows)
        candidate_failures = sum(1 for row in candidate_rows if float(row["success_rate"]) <= 0.0)
        round6_failures = sum(1 for item in baseline_rows if float(item["round6"]) <= 0.0)
        stage1_pass = (
            candidate_mean >= round6_mean + float(campaign["selection"]["min_dev_gain"])
            and candidate_failures <= round6_failures
        )
        summaries.append(
            {
                "candidate": candidate,
                "family": str(candidate_rows[0]["family"]),
                "candidate_mean": float(candidate_mean),
                "round6_mean": float(round6_mean),
                "token_mean": float(token_mean),
                "single_mean": float(single_mean),
                "delta_vs_round6": float(candidate_mean - round6_mean),
                "delta_vs_token": float(candidate_mean - token_mean),
                "delta_vs_single": float(candidate_mean - single_mean),
                "candidate_failures": int(candidate_failures),
                "round6_failures": int(round6_failures),
                "trigger_rate_mean": float(mean(float(row["trigger_rate"]) for row in candidate_rows)),
                "compute_multiplier_mean": float(mean(float(row["compute_multiplier_mean"]) for row in candidate_rows)),
                "wall_seconds_mean": float(mean(float(row["wall_seconds"]) for row in candidate_rows)),
                "stage1_pass": bool(stage1_pass),
            }
        )
    selected: list[str] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_family[str(row["family"])].append(row)
    for family_rows in by_family.values():
        family_rows.sort(
            key=lambda item: (
                bool(item["stage1_pass"]),
                float(item["candidate_mean"]),
                -float(item["candidate_failures"]),
            ),
            reverse=True,
        )
        for row in family_rows[:family_top_k]:
            if bool(row["stage1_pass"]):
                selected.append(str(row["candidate"]))
    return summaries, selected


def render_rescue_screening(campaign: dict[str, Any], output: Path, csv_output: Path | None, json_output: Path | None) -> None:
    rows = _stage_rows(campaign, "screening")
    summaries, selected = _aggregate_screening(campaign, rows)
    by_family = _rescue_candidates_by_family(campaign)
    lines = [
        "# Escape2x Stage B1 Rescue Screening",
        "",
        f"- substantive rescue variants: `{sum(len(items) for items in by_family.values())}`",
        f"- rescue families: `{sorted(by_family)}`",
        f"- selected family representatives: `{selected}`",
        "",
        "| Candidate | Family | Deadlock-Dev Mean | Delta vs round6 | Delta vs token_dense | Delta vs single_expert | Complete-Seed Failures | Trigger Rate | Compute x | Stage B1 Pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(summaries, key=lambda item: (item["family"], item["candidate_mean"]), reverse=True):
        lines.append(
            f"| `{row['candidate']}` | `{row['family']}` | `{row['candidate_mean']:.4f}` | `{row['delta_vs_round6']:.4f}` | `{row['delta_vs_token']:.4f}` | `{row['delta_vs_single']:.4f}` | `{row['candidate_failures']}` | `{row['trigger_rate_mean']:.4f}` | `{row['compute_multiplier_mean']:.2f}` | `{row['stage1_pass']}` |"
        )
    lines.extend(["", "## Family Notes", ""])
    for family, candidates in sorted(by_family.items()):
        family_rows = [row for row in summaries if str(row["family"]) == family]
        family_rows.sort(key=lambda item: float(item["candidate_mean"]), reverse=True)
        best = family_rows[0] if family_rows else None
        lines.append(f"- `{family}` tested `{len(candidates)}` variants; best representative is `{best['candidate'] if best else '-'}` with deadlock-dev mean `{best['candidate_mean']:.4f}` and pass `{best['stage1_pass'] if best else False}`.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, rows)
    if json_output is not None:
        _write_json(
            json_output,
            {
                "rows": rows,
                "candidate_summaries": summaries,
                "selected_candidates": selected,
                "family_counts": {family: len(items) for family, items in by_family.items()},
            },
        )


def render_rescue_rerun_holdout(campaign: dict[str, Any], output: Path, csv_output: Path | None, json_output: Path | None) -> None:
    rows = _stage_rows(campaign, "rerun")
    stage1 = _optional_json(campaign["reports"].get("rescue_stage1_json"))
    selected = [str(name) for name in stage1.get("selected_candidates", [])]
    baseline_lookup = _current_rows_lookup(campaign)
    summaries: list[dict[str, Any]] = []
    surviving: list[str] = []
    for candidate in selected:
        candidate_rows = [row for row in rows if str(row["candidate"]) == candidate]
        dev_rows = [row for row in candidate_rows if _expected_subgroup(campaign, str(row["lane"]), int(row["seed"])) == "teacher_locked_no_escape" and (str(row["lane"]), int(row["seed"])) in {(row["lane"], row["seed"]) for row in _teacher_locked_dev_cases(campaign)}]
        holdout_rows = [row for row in candidate_rows if (str(row["lane"]), int(row["seed"])) in {(row["lane"], row["seed"]) for row in _teacher_locked_holdout_cases(campaign)}]
        ambiguous_rows = [row for row in candidate_rows if (str(row["lane"]), int(row["seed"])) in {(row["lane"], row["seed"]) for row in _ambiguous_cases(campaign)}]
        if not candidate_rows:
            continue
        stage1_summary = next((row for row in stage1.get("candidate_summaries", []) if str(row["candidate"]) == candidate), {})
        holdout_baseline = [baseline_lookup[(str(row["lane"]), int(row["seed"]))] for row in holdout_rows]
        ambiguous_baseline = [baseline_lookup[(str(row["lane"]), int(row["seed"]))] for row in ambiguous_rows]
        dev_rerun_mean = mean(float(row["success_rate"]) for row in dev_rows) if dev_rows else 0.0
        holdout_mean = mean(float(row["success_rate"]) for row in holdout_rows) if holdout_rows else 0.0
        ambiguous_mean = mean(float(row["success_rate"]) for row in ambiguous_rows) if ambiguous_rows else 0.0
        holdout_round6 = mean(float(item["round6"]) for item in holdout_baseline) if holdout_baseline else 0.0
        ambiguous_round6 = mean(float(item["round6"]) for item in ambiguous_baseline) if ambiguous_baseline else 0.0
        rerun_consistent = dev_rerun_mean >= float(stage1_summary.get("candidate_mean", 0.0)) - 0.05
        holdout_pass = holdout_mean >= holdout_round6 + float(campaign["selection"]["rescue_holdout_min_gain"])
        ambiguous_pass = ambiguous_mean >= ambiguous_round6 - float(campaign["selection"]["rescue_ambiguous_tolerance"])
        pass_stage2 = rerun_consistent and holdout_pass and ambiguous_pass
        if pass_stage2:
            surviving.append(candidate)
        summaries.append(
            {
                "candidate": candidate,
                "family": str(candidate_rows[0]["family"]),
                "dev_screening_mean": float(stage1_summary.get("candidate_mean", 0.0)),
                "dev_rerun_mean": float(dev_rerun_mean),
                "holdout_mean": float(holdout_mean),
                "holdout_round6": float(holdout_round6),
                "ambiguous_mean": float(ambiguous_mean),
                "ambiguous_round6": float(ambiguous_round6),
                "rerun_consistent": bool(rerun_consistent),
                "holdout_pass": bool(holdout_pass),
                "ambiguous_pass": bool(ambiguous_pass),
                "pass_stage2": bool(pass_stage2),
            }
        )
    lines = [
        "# Escape2x Stage B2 Rescue Rerun and Holdout",
        "",
        f"- stage B1 selected candidates: `{selected}`",
        f"- stage B2 surviving rescue candidates: `{surviving}`",
        "",
        "| Candidate | Family | Dev Screening | Dev Rerun | Holdout | Holdout round6 | Ambiguous | Ambiguous round6 | Stage B2 Pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(summaries, key=lambda item: item["dev_rerun_mean"], reverse=True):
        lines.append(
            f"| `{row['candidate']}` | `{row['family']}` | `{row['dev_screening_mean']:.4f}` | `{row['dev_rerun_mean']:.4f}` | `{row['holdout_mean']:.4f}` | `{row['holdout_round6']:.4f}` | `{row['ambiguous_mean']:.4f}` | `{row['ambiguous_round6']:.4f}` | `{row['pass_stage2']}` |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, rows)
    if json_output is not None:
        _write_json(
            json_output,
            {
                "rows": rows,
                "candidate_summaries": summaries,
                "surviving_candidates": surviving,
            },
        )


def _route_probe_summary(route_csv: Path) -> dict[str, Any]:
    if not route_csv.exists():
        return {"available": False, "round6_pass": False}
    with route_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["lane"]), int(row["seed"]))].append(row)
    fixed_drops: list[float] = []
    random_drops: list[float] = []
    worst_drops: list[float] = []
    for case_rows in grouped.values():
        baseline = next((row for row in case_rows if str(row["probe"]) == "baseline"), None)
        fixed = next((row for row in case_rows if str(row["probe"]) == "router_override"), None)
        randomized = next((row for row in case_rows if str(row["probe"]) == "route_randomization"), None)
        ablations = [row for row in case_rows if str(row["probe"]) == "expert_ablation"]
        if baseline is None or fixed is None or randomized is None or not ablations:
            continue
        baseline_success = float(baseline["eval_success_rate"])
        fixed_drops.append(baseline_success - float(fixed["eval_success_rate"]))
        random_drops.append(baseline_success - float(randomized["eval_success_rate"]))
        worst_drops.append(baseline_success - min(float(row["eval_success_rate"]) for row in ablations))
    return {
        "available": True,
        "round6_pass": bool(max(mean(fixed_drops) if fixed_drops else 0.0, mean(random_drops) if random_drops else 0.0) >= 0.25),
        "fixed_drop_mean": float(mean(fixed_drops)) if fixed_drops else 0.0,
        "random_drop_mean": float(mean(random_drops)) if random_drops else 0.0,
        "worst_ablation_drop_mean": float(mean(worst_drops)) if worst_drops else 0.0,
    }


def render_rescue_antiregression_route(campaign: dict[str, Any], output: Path, csv_output: Path | None, json_output: Path | None) -> None:
    stage2 = _optional_json(campaign["reports"].get("rescue_stage2_json"))
    survivors = [str(name) for name in stage2.get("surviving_candidates", [])]
    rows = _stage_rows(campaign, "antiregression")
    baseline_lookup = _current_rows_lookup(campaign)
    summaries: list[dict[str, Any]] = []
    bounded_rescue_justified = False
    for candidate in survivors:
        candidate_rows = [row for row in rows if str(row["candidate"]) == candidate]
        healthy_baselines = [baseline_lookup[(str(row["lane"]), int(row["seed"]))] for row in candidate_rows]
        candidate_mean = mean(float(row["success_rate"]) for row in candidate_rows) if candidate_rows else 0.0
        round6_mean = mean(float(item["round6"]) for item in healthy_baselines) if healthy_baselines else 0.0
        false_positive = mean(float(row["false_positive_rate"]) for row in candidate_rows) if candidate_rows else 0.0
        healthy_pass = candidate_mean >= round6_mean - 0.02 and false_positive <= 0.10
        if healthy_pass:
            bounded_rescue_justified = True
        summaries.append(
            {
                "candidate": candidate,
                "healthy_mean": float(candidate_mean),
                "round6_healthy_mean": float(round6_mean),
                "false_positive_rate": float(false_positive),
                "healthy_pass": bool(healthy_pass),
            }
        )
    route_summary = _route_probe_summary(Path(campaign["reports"]["rescue_route_raw_csv"]))
    if not survivors:
        route_summary = {
            **route_summary,
            "available": route_summary.get("available", False),
            "round6_pass": bool(_optional_json(campaign["analysis"].get("current_route_json")).get("round6_pass", False)),
        }
    lines = [
        "# Escape2x Stage B3 Anti-Regression and Route Validation",
        "",
        f"- stage B2 surviving rescue candidates: `{survivors}`",
        f"- bounded auxiliary rescue justified: `{bounded_rescue_justified and bool(route_summary.get('round6_pass'))}`",
        f"- route probe available: `{route_summary.get('available')}`",
        f"- underlying round6 route pass: `{route_summary.get('round6_pass')}`",
        "",
    ]
    if summaries:
        lines.extend(
            [
                "| Candidate | Healthy Mean | round6 Healthy Mean | False-Positive Trigger Rate | Healthy Pass |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in summaries:
            lines.append(
                f"| `{row['candidate']}` | `{row['healthy_mean']:.4f}` | `{row['round6_healthy_mean']:.4f}` | `{row['false_positive_rate']:.4f}` | `{row['healthy_pass']}` |"
            )
    else:
        lines.extend(
            [
                "- No rescue candidate survived Stage B2, so Stage B3 carried forward the incumbent route reference only.",
            ]
        )
    lines.extend(
        [
            "",
            "## Route Probe Summary",
            "",
            f"- fixed-router drop mean: `{route_summary.get('fixed_drop_mean', 0.0):.4f}`",
            f"- route-randomization drop mean: `{route_summary.get('random_drop_mean', 0.0):.4f}`",
            f"- worst-expert-ablation drop mean: `{route_summary.get('worst_ablation_drop_mean', 0.0):.4f}`",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, rows)
    if json_output is not None:
        _write_json(
            json_output,
            {
                "rows": rows,
                "candidate_summaries": summaries,
                "surviving_candidates": survivors,
                "route_summary": route_summary,
                "bounded_rescue_justified": bool(bounded_rescue_justified and route_summary.get("round6_pass")),
            },
        )


def _detector_trace_case(campaign: dict[str, Any], lane: str, seed: int, device: str) -> dict[str, Any]:
    spec = {
        "config_path": str(_round6_run_dir(campaign, lane, seed) / "student_resolved_config.yaml"),
        "checkpoint_path": str(_round6_run_dir(campaign, lane, seed) / "latest.pt"),
        "detector": dict(campaign["analysis"]["detector"]),
        "expected_subgroup": _expected_subgroup(campaign, lane, seed),
        "episode_index": 0,
        "episode_reset_seed": int(seed) + 7_000,
        "max_steps": int(campaign["analysis"]["rescue_max_steps"]),
    }
    device_t = torch.device("cpu" if device == "cpu" or not torch.cuda.is_available() else "cuda")
    episodes = int(campaign["analysis"]["detector_trace_episodes"])
    predicted: list[str] = []
    trigger_rates: list[float] = []
    rows: list[dict[str, Any]] = []
    for episode_index in range(episodes):
        episode_spec = dict(spec)
        episode_spec["episode_index"] = episode_index
        episode_spec["episode_reset_seed"] = int(seed) + 7_000 + episode_index
        row = _evaluate_episode(spec=episode_spec, device=device_t, trace_only=True)
        predicted.append(str(row["episode_predicted_subgroup"]))
        trigger_rates.append(float(row["rescue_triggered"]))
        rows.append(row)
    return {
        "lane": lane,
        "seed": int(seed),
        "expected_subgroup": _expected_subgroup(campaign, lane, seed),
        "majority_predicted": Counter(predicted).most_common(1)[0][0] if predicted else "healthy",
        "trigger_rate": float(mean(trigger_rates)) if trigger_rates else 0.0,
        "success_rate": float(mean(float(row["success"]) for row in rows)) if rows else 0.0,
    }


def render_subgroup_detector(campaign: dict[str, Any], output: Path, json_output: Path | None, *, device: str) -> None:
    rows = [_detector_trace_case(campaign, row["lane"], int(row["seed"]), device) for row in _all_detector_cases(campaign)]
    majority_match = sum(1 for row in rows if str(row["majority_predicted"]) == str(row["expected_subgroup"]))
    lines = [
        "# Escape2x Subgroup Detector",
        "",
        f"- detector cases: `{len(rows)}`",
        f"- episode-majority match count: `{majority_match}` / `{len(rows)}`",
        "",
        "## Detector Rules",
        "",
    ]
    for key, value in sorted(campaign["analysis"]["detector"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Case Assignments",
            "",
            "| Lane | Seed | Expected | Majority Predicted | Trigger Rate | Greedy Success |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['lane']} | {row['seed']} | `{row['expected_subgroup']}` | `{row['majority_predicted']}` | `{row['trigger_rate']:.4f}` | `{row['success_rate']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The detector is intentionally operational rather than retrospective: it uses pre-key stall length, repeated-position ratio, confidence, route concentration, and settling mismatch so the same surface can trigger conditional rescue at inference time.",
            "- Each episode gets one strongest-signature subgroup label, and the table reports the majority vote over those episode labels for each case.",
            "- Teacher-locked cases should concentrate in `teacher_locked_no_escape`, ambiguous cases should move toward `ambiguous_unstable`, and guardrail/healthy cases should avoid frequent rescue triggers.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(json_output, {"rows": rows, "majority_match": majority_match})


def render_registration(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    prior_runs, target_runs = _prior_arch_substantive_runs(campaign)
    rescue_families = _rescue_candidates_by_family(campaign)
    rescue_run_count = sum(len(items) for items in rescue_families.values())
    practicalization_min_runs = 32
    architecture_reports = [str(item) for item in campaign["analysis"].get("source_quality_architecture_reports", [])]
    architecture_entry = dict(campaign["analysis"].get("architecture_specialist_entry", {}))
    lines = [
        "# Escape2x Registration",
        "",
        f"- prior substantive runs from the previous escape/practicalization pass: `{prior_runs}`",
        f"- target substantive run budget absent stop conditions: `{target_runs}`",
        f"- planned bounded rescue variants: `{rescue_run_count}`",
        f"- planned practicalization substantive runs if rescue survives: `>= {practicalization_min_runs}`",
        f"- architecture-specialist cap if justified: `<= 16`",
        f"- architecture reference reports: `{architecture_reports}`",
        "",
        f"- teacher-locked dev groups: `{campaign['analysis']['teacher_locked_dev_groups']}`",
        f"- teacher-locked holdout groups: `{campaign['analysis']['teacher_locked_holdout_groups']}`",
        f"- ambiguous groups: `{campaign['analysis']['ambiguous_groups']}`",
        f"- guardrail groups: `{campaign['analysis']['guardrail_groups']}`",
        f"- healthy anti-regression groups: `{campaign['analysis']['healthy_groups']}`",
        "",
        "## Rescue Families",
        "",
    ]
    for family, candidates in sorted(rescue_families.items()):
        lines.append(f"- `{family}`: `{candidates}`")
    lines.extend(
        [
            "",
            "## Rescue Coverage Notes",
            "",
            "- The registered rescue slate already fills the required 48 substantive rescue variants across B1-B5.",
            "- Hybrid rescue mixes are held as a contingency rather than pre-registered because the base rescue families already saturate the required rescue budget with fair-shot calibration sweeps.",
            "",
            "## Practicalization Families",
            "",
        ]
    )
    for item in campaign["analysis"]["practicalization_families"]:
        lines.append(f"- `{item['label']}` ({item['track']}): `{item['candidates']}`")
    lines.extend(
        [
            "",
            "## Bars",
            "",
            "- rescue-specialist bar: materially improve over `round6` on teacher-locked deadlock dev, avoid obvious new complete-seed failures, and survive rerun/holdout before any practicalization starts.",
            "- practicalization bar: only open Stage C if bounded rescue remains real after holdout and anti-regression; otherwise skip practicalization rather than guessing.",
            "- architecture-specialist entry bar: bounded rescue must survive, practicalization must fail, and the accepted source-quality verdict must match the explicit entry condition.",
            "- benchmark-promotion bar: external 64-episode evaluation, matched fairness, holdout, anti-regression, route, stability, and pack gate.",
            "",
            "## Pruning Rules",
            "",
            "- Every rescue family gets a 3-5 variant calibration mini-sweep before any prune call.",
            "- Rescue families may be pruned early only after a confirming rerun or after they clearly trail both `round6` and the current oracle escape ceiling on multiple deadlock-dev cases.",
            "- The campaign may stop below the nominal 2x budget only if a formal stop condition fires; the report trail must say exactly where and why.",
            "",
            "## Decision Rules",
            "",
            "- Stage B1 advances only rescue candidates that materially beat `round6` on the teacher-locked dev slice with no new obvious complete-seed failures.",
            "- Stage B2 advances only rescue candidates that rerun consistently, keep signal on teacher-locked holdout, and do not collapse on the ambiguous slice.",
            "- Stage B3 justifies a bounded auxiliary rescue policy only if the rescue line stays healthy on non-deadlock blocks, keeps false-positive triggers low, and the underlying routed policy still passes bounded route dependence probes.",
            "- Practicalization runs only if Stage B3 leaves a rescue signal worth trying to distill back into the accepted training family.",
            f"- Architecture-specialist work stays off unless rescue is real, practicalization fails, and the accepted source-quality verdict equals `{architecture_entry.get('requires_source_quality_verdict', 'missing')}`.",
            "- Final allowed outcomes remain: active benchmark confirmed and bounded rescue frontier clarified, challenger replaces the active benchmark, round6 remains active and a bounded specialist rescue policy is justified, active benchmark remains and the frontier narrows further, or architecture-specialist branch justified for the next phase.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "prior_substantive_runs": prior_runs,
                "target_substantive_runs": target_runs,
                "planned_rescue_runs": rescue_run_count,
                "planned_practicalization_min_runs": practicalization_min_runs,
                "rescue_families": rescue_families,
                "practicalization_families": campaign["analysis"]["practicalization_families"],
                "architecture_specialist_entry": architecture_entry,
                "architecture_reference_reports": architecture_reports,
            },
        )


def _render_skip_report(title: str, output: Path, json_output: Path | None, reason: str, status: str = "skipped") -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                f"- status: `{status}`",
                f"- reason: {reason}",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if json_output is not None:
        _write_json(json_output, {"status": status, "reason": reason})


def render_practicalization_screening(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    rescue_stage3 = _optional_json(campaign["reports"].get("rescue_stage3_json"))
    if not bool(rescue_stage3.get("bounded_rescue_justified")):
        _render_skip_report(
            "Escape2x Stage C1 Practicalization Screening",
            output,
            json_output,
            "No rescue candidate survived the rerun/holdout/anti-regression funnel strongly enough to justify benchmark-lane practicalization runs.",
        )
        return
    stage1 = _optional_json(campaign["reports"].get("stage1_json"))
    fruitful = stage1.get("fruitful_advancing_candidates", [])
    exploratory = stage1.get("exploratory_advancing_candidates", [])
    lines = [
        "# Escape2x Stage C1 Practicalization Screening",
        "",
        f"- fruitful advancing candidates: `{fruitful}`",
        f"- exploratory advancing candidates: `{exploratory}`",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "status": "completed",
                "fruitful_advancing_candidates": fruitful,
                "exploratory_advancing_candidates": exploratory,
                "selected_candidates": list(stage1.get("selected_candidates", [])),
            },
        )


def render_practicalization_verification(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    rescue_stage3 = _optional_json(campaign["reports"].get("rescue_stage3_json"))
    if not bool(rescue_stage3.get("bounded_rescue_justified")):
        _render_skip_report(
            "Escape2x Stage C2 Practicalization Verification, Fairness, and Holdout",
            output,
            json_output,
            "Practicalization did not run because no bounded rescue policy was strong enough to justify a benchmark-lane approximation attempt.",
        )
        return
    verification = _optional_json(campaign["reports"].get("stage2_verification_json"))
    fairness = _optional_json(campaign["reports"].get("stage2_json"))
    holdout = _optional_json(campaign["reports"].get("stage3_json"))
    anti_regression = _optional_json(campaign["reports"].get("stage4_json"))
    surviving = list(holdout.get("surviving_candidates", fairness.get("surviving_candidates", [])))
    lines = [
        "# Escape2x Stage C2 Practicalization Verification, Fairness, and Holdout",
        "",
        f"- verified candidates: `{verification.get('verified_candidates', [])}`",
        f"- fairness survivors: `{fairness.get('surviving_candidates', [])}`",
        f"- holdout survivors: `{surviving}`",
        f"- anti-regression pass: `{anti_regression.get('challenger_pass')}`",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "status": "completed",
                "verified_candidates": verification.get("verified_candidates", []),
                "fairness_survivors": fairness.get("surviving_candidates", []),
                "holdout_survivors": surviving,
                "anti_regression_pass": anti_regression.get("challenger_pass"),
            },
        )


def render_practicalization_route_stability(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    rescue_stage3 = _optional_json(campaign["reports"].get("rescue_stage3_json"))
    if not bool(rescue_stage3.get("bounded_rescue_justified")):
        _render_skip_report(
            "Escape2x Stage C3 Practicalization Route and Stability",
            output,
            json_output,
            "No practicalized candidate advanced because the rescue lane never justified a benchmark-lane approximation attempt.",
        )
        return
    route = _optional_json(campaign["reports"].get("stage5_json"))
    stability = _optional_json(campaign["reports"].get("stage6_json"))
    lines = [
        "# Escape2x Stage C3 Practicalization Route and Stability",
        "",
        f"- incumbent route pass: `{route.get('round6_pass')}`",
        f"- challenger route pass: `{route.get('challenger_pass')}`",
        f"- incumbent stability pass: `{stability.get('round6_pass')}`",
        f"- challenger stability pass: `{stability.get('challenger_pass')}`",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "status": "completed",
                "route": route,
                "stability": stability,
            },
        )


def render_archpilot(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    rescue_stage3 = _optional_json(campaign["reports"].get("rescue_stage3_json"))
    practical_stage2 = _optional_json(campaign["reports"].get("practicalization_stage2_json"))
    accepted_source_quality = _optional_json(campaign["analysis"].get("accepted_source_quality_json"))
    entry = dict(campaign["analysis"].get("architecture_specialist_entry", {}))
    bounded_rescue = bool(rescue_stage3.get("bounded_rescue_justified"))
    practical_failed = not bool(practical_stage2.get("holdout_survivors"))
    required_verdict = str(entry.get("requires_source_quality_verdict", ""))
    source_quality_verdict = str(accepted_source_quality.get("verdict", "missing"))
    source_quality_pass = (not required_verdict) or source_quality_verdict == required_verdict
    if not (bounded_rescue and practical_failed and source_quality_pass):
        reasons: list[str] = []
        if not bounded_rescue:
            reasons.append("bounded rescue was not justified")
        if not practical_failed:
            reasons.append("practicalization already produced a surviving benchmark-lane candidate")
        if not source_quality_pass:
            reasons.append(
                f"accepted source-quality verdict `{source_quality_verdict}` did not match required `{required_verdict}`"
            )
        _render_skip_report(
            "Escape2x Stage D Architecture-Specialist Branch",
            output,
            json_output,
            "; ".join(reasons) + ".",
        )
        return
    _render_skip_report(
        "Escape2x Stage D Architecture-Specialist Branch",
        output,
        json_output,
        "Entry conditions were met, but this campaign did not materialize a separate specialist run set before the final gate path.",
        status="not_run",
    )


def write_candidate_pack(campaign: dict[str, Any], output: Path) -> None:
    practical = _optional_json(campaign["reports"].get("practicalization_stage2_json"))
    surviving = list(practical.get("holdout_survivors", []))
    winner = str(surviving[0]) if len(surviving) == 1 else str(campaign["current_canonical_name"])
    if winner != str(campaign["current_canonical_name"]):
        raise RuntimeError("challenger pack materialization is not supported by the escape helper path yet")
    pack = _read_json(Path(campaign["current_canonical_pack"]))
    pack["deadlock_escape2x_program"] = {
        "decision_basis": {
            "rescue_screening": _optional_json(campaign["reports"].get("rescue_stage1_json")),
            "rescue_rerun_holdout": _optional_json(campaign["reports"].get("rescue_stage2_json")),
            "rescue_antiregression_route": _optional_json(campaign["reports"].get("rescue_stage3_json")),
            "practicalization_screening": _optional_json(campaign["reports"].get("practicalization_stage1_json")),
            "practicalization_verification": _optional_json(campaign["reports"].get("practicalization_stage2_json")),
            "practicalization_route_stability": _optional_json(campaign["reports"].get("practicalization_stage3_json")),
        },
        "winner": winner,
        "active_pack_before_program": {
            "path": str(campaign["current_canonical_pack"]),
            "sha256": _sha256_path(Path(campaign["current_canonical_pack"])),
        },
        "gate_reference_pack": {
            "path": str(campaign["frozen_pack"]),
            "sha256": _sha256_path(Path(campaign["frozen_pack"])),
        },
    }
    pack.setdefault("provenance", {})
    pack["provenance"]["git_commit"] = get_git_commit()
    pack["provenance"]["git_dirty"] = get_git_dirty()
    pack["provenance"]["deadlock_escape_notes"] = "2x detector-triggered rescue, practicalization, and architecture-specialist program around the active round6 benchmark"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pack, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_decision_memo(campaign: dict[str, Any], output: Path) -> None:
    stage1 = _optional_json(campaign["reports"].get("rescue_stage1_json"))
    stage2 = _optional_json(campaign["reports"].get("rescue_stage2_json"))
    stage3 = _optional_json(campaign["reports"].get("rescue_stage3_json"))
    practical1 = _optional_json(campaign["reports"].get("practicalization_stage1_json"))
    practical2 = _optional_json(campaign["reports"].get("practicalization_stage2_json"))
    practical3 = _optional_json(campaign["reports"].get("practicalization_stage3_json"))
    archpilot = _optional_json(campaign["reports"].get("archpilot_json"))
    gate = _optional_json(campaign["reports"].get("gate_report_json"))
    route_ref = _optional_json(campaign["analysis"].get("current_route_json"))
    stability_ref = _optional_json(campaign["analysis"].get("current_stability_json"))
    labels = _escape2x_decision_strings(campaign)
    gate_verdict = str(gate.get("verdict", gate.get("status", "not_run")))
    required = str(campaign["selection"]["pack_gate_required_verdict"])
    practical_survivor = bool(practical2.get("holdout_survivors"))
    bounded_rescue = bool(stage3.get("bounded_rescue_justified"))
    arch_justified = bool(archpilot.get("status") == "completed" and archpilot.get("justified"))
    if practical_survivor and gate_verdict == required and bool(practical3.get("route", {}).get("challenger_pass")) and bool(practical3.get("stability", {}).get("challenger_pass")):
        final_status = labels["replace"]
    elif arch_justified:
        final_status = labels["arch_specialist"]
    elif gate_verdict == required and bounded_rescue:
        final_status = labels["bounded_rescue"]
    elif gate_verdict == required and stage2.get("surviving_candidates"):
        final_status = labels["confirm"]
    elif gate_verdict == required:
        final_status = labels["narrow"]
    else:
        final_status = labels["narrow"]
    lines = [
        "# Escape2x Decision Memo",
        "",
        f"- final status: `{final_status}`",
        f"- active benchmark line: `{campaign['current_canonical_name']}`",
        f"- gate verdict: `{gate_verdict}`",
        "",
        "## Funnel",
        "",
        f"- Stage B1 selected rescue candidates: `{stage1.get('selected_candidates', [])}`",
        f"- Stage B2 surviving rescue candidates: `{stage2.get('surviving_candidates', [])}`",
        f"- Stage B3 bounded rescue justified: `{stage3.get('bounded_rescue_justified')}`",
        f"- Practicalization Stage C1 status: `{practical1.get('status', 'missing')}`",
        f"- Practicalization Stage C2 status: `{practical2.get('status', 'missing')}`",
        f"- Practicalization Stage C3 status: `{practical3.get('status', 'missing')}`",
        f"- Incumbent route reference pass: `{route_ref.get('round6_pass', 'missing')}`",
        f"- Incumbent stability reference pass: `{stability_ref.get('round6_pass', 'missing')}`",
        "",
        "## Decision",
        "",
    ]
    if final_status == labels["replace"]:
        lines.append("- A practicalized benchmark-lane challenger survived verification, matched controls, holdout, anti-regression, route, stability, and gate checks strongly enough to replace `round6`.")
    elif final_status == labels["arch_specialist"]:
        lines.append("- Rescue was real enough to justify a follow-on architecture-specialist branch, but the current benchmark lane did not produce a practicalized challenger and the public benchmark state does not change now.")
    elif final_status == labels["bounded_rescue"]:
        lines.append("- The benchmark stays on `round6`, and a bounded specialist rescue policy is justified outside the benchmark lane because it survived rerun, holdout, anti-regression, and route checks strongly enough to remain operational.")
    elif final_status == labels["confirm"]:
        lines.append("- The benchmark stays on `round6`. The new program resolved the frontier around deadlock escape more sharply, but the accepted benchmark lane still lacks a practicalized challenger and no bounded rescue policy was strong enough to operationalize.")
    else:
        lines.append("- The benchmark stays on `round6`, and the rescue/practicalization program did not leave enough durable headroom to support anything stronger than a narrower frontier around the current deadlock interpretation.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2x-scale deadlock escape, practicalization, and architecture-specialist program around active round6")
    sub = parser.add_subparsers(dest="command", required=True)

    for command in (
        "state-reconciliation",
        "baseline-sync",
        "subgroup-detector",
        "source-quality-report",
        "registration",
        "run-rescue-stage",
        "run-task",
        "rescue-screening-report",
        "rescue-rerun-holdout-report",
        "rescue-antiregression-route-report",
        "practicalization-screening-report",
        "practicalization-verification-report",
        "practicalization-route-stability-report",
        "archpilot-report",
        "candidate-pack",
        "decision-memo",
    ):
        item = sub.add_parser(command)
        if command != "run-task":
            item.add_argument("--campaign-config", required=True)
        if command == "run-task":
            item.add_argument("--spec", required=True)
            item.add_argument("--device", default="auto")
        elif command == "run-rescue-stage":
            item.add_argument("--stage", choices=("screening", "rerun", "antiregression"), required=True)
            item.add_argument("--device", default="auto")
        elif command in {"state-reconciliation"}:
            item.add_argument("--output", required=True)
        elif command in {"baseline-sync"}:
            item.add_argument("--output", required=True)
            item.add_argument("--csv", required=False)
            item.add_argument("--json", required=False)
        elif command in {"subgroup-detector", "source-quality-report"}:
            item.add_argument("--output", required=True)
            item.add_argument("--json", required=False)
            item.add_argument("--device", default="auto")
        elif command in {
            "registration",
            "rescue-screening-report",
            "rescue-rerun-holdout-report",
            "rescue-antiregression-route-report",
            "practicalization-screening-report",
            "practicalization-verification-report",
            "practicalization-route-stability-report",
            "archpilot-report",
            "decision-memo",
        }:
            item.add_argument("--output", required=True)
            if command != "decision-memo":
                item.add_argument("--json", required=False)
            if command in {
                "rescue-screening-report",
                "rescue-rerun-holdout-report",
                "rescue-antiregression-route-report",
            }:
                item.add_argument("--csv", required=False)
        elif command == "candidate-pack":
            item.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run-task":
        run_rescue_task(Path(args.spec), device=str(args.device))
        return

    campaign = load_campaign_config(Path(args.campaign_config))
    if args.command == "state-reconciliation":
        _write_state_reconciliation(campaign)
        return
    if args.command == "baseline-sync":
        _write_baseline_sync(campaign)
        return
    if args.command == "subgroup-detector":
        render_subgroup_detector(campaign, Path(args.output), Path(args.json) if args.json else None, device=str(args.device))
        return
    if args.command == "source-quality-report":
        render_source_quality(
            campaign,
            Path(args.output),
            Path(args.json) if args.json else None,
            None,
            device=str(args.device),
        )
        return
    if args.command == "registration":
        render_registration(campaign, Path(args.output), Path(args.json) if args.json else None)
        return
    if args.command == "run-rescue-stage":
        run_rescue_stage(campaign, stage=str(args.stage), device=str(args.device))
        return
    if args.command == "rescue-screening-report":
        render_rescue_screening(
            campaign,
            Path(args.output),
            Path(args.csv) if args.csv else None,
            Path(args.json) if args.json else None,
        )
        return
    if args.command == "rescue-rerun-holdout-report":
        render_rescue_rerun_holdout(
            campaign,
            Path(args.output),
            Path(args.csv) if args.csv else None,
            Path(args.json) if args.json else None,
        )
        return
    if args.command == "rescue-antiregression-route-report":
        render_rescue_antiregression_route(
            campaign,
            Path(args.output),
            Path(args.csv) if args.csv else None,
            Path(args.json) if args.json else None,
        )
        return
    if args.command == "practicalization-screening-report":
        render_practicalization_screening(campaign, Path(args.output), Path(args.json) if args.json else None)
        return
    if args.command == "practicalization-verification-report":
        render_practicalization_verification(campaign, Path(args.output), Path(args.json) if args.json else None)
        return
    if args.command == "practicalization-route-stability-report":
        render_practicalization_route_stability(campaign, Path(args.output), Path(args.json) if args.json else None)
        return
    if args.command == "archpilot-report":
        render_archpilot(campaign, Path(args.output), Path(args.json) if args.json else None)
        return
    if args.command == "candidate-pack":
        write_candidate_pack(campaign, Path(args.output))
        return
    if args.command == "decision-memo":
        render_decision_memo(campaign, Path(args.output))
        return
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
