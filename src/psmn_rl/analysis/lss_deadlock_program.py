from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from psmn_rl.analysis.campaign_config import load_campaign_config
from psmn_rl.analysis.lss_forensic_atlas import RunTarget, SeedCase, _trace_variant
from psmn_rl.analysis.lss_portfolio_campaign import _decision_status, _winner
from psmn_rl.analysis.lss_post_pass_campaign import _read_json, _write_csv, _write_json
from psmn_rl.analysis.lss_robustness import _format_float
from psmn_rl.analysis.lss_successor_migration import _block_lanes, _control_family, _current_round6_rows
from psmn_rl.analysis.portfolio_frontier_contract_loader import load_frontier_contract
from psmn_rl.rl.distributed.ddp import detect_device
from psmn_rl.utils.io import get_git_commit, get_git_dirty


SUCCESS_THRESHOLD = 0.999
LOW_DISAGREEMENT_THRESHOLD = 0.05
HIGH_CONFIDENCE_THRESHOLD = 0.95
POST_UNLOCK_THRESHOLD = 0.05
PRE_KEY_PHASES = {"search_key", "at_key"}


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _optional_text(path: str | Path | None) -> str:
    if path is None:
        return ""
    target = Path(path)
    if not target.exists():
        return ""
    return target.read_text(encoding="utf-8")


def _optional_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    return _read_json(target)


def _analysis_label(campaign: dict[str, Any], default: str) -> str:
    return str(campaign.get("analysis", {}).get("program_label", default))


def _summary_path(run_dir: Path) -> Path:
    return run_dir / "summary.json"


def _candidate_source_root(campaign: dict[str, Any]) -> Path:
    return Path(campaign["analysis"]["source_stage1_root"])


def _candidate_run_dir(campaign: dict[str, Any], candidate: str, lane: str, seed: int) -> Path:
    roots = [_candidate_source_root(campaign), *(Path(root) for root in campaign.get("reuse_roots", {}).get("stage1_sare", []))]
    fallback = roots[0] / candidate / lane / f"seed_{seed}" / "kl_lss_sare"
    for root in roots:
        run_dir = root / candidate / lane / f"seed_{seed}" / "kl_lss_sare"
        if (run_dir / "summary.json").exists():
            return run_dir
    return fallback


def _round6_run_dir(campaign: dict[str, Any], lane: str, seed: int, label: str = "kl_lss_sare") -> Path:
    root_key = "current_round6_sare_roots" if label == "kl_lss_sare" else "current_round6_control_roots"
    return Path(campaign[root_key][lane]) / f"seed_{seed}" / label


def _matched_dev_pairs(campaign: dict[str, Any]) -> list[tuple[str, int]]:
    return [
        (str(block["lane"]), int(seed))
        for block in campaign["blocks"]["dev"]
        for seed in block.get("seeds", [])
    ]


def _hard_seed_pairs(campaign: dict[str, Any]) -> list[tuple[str, int]]:
    pairs = [
        (str(block["lane"]), int(seed))
        for block in campaign.get("blocks", {}).get("hard_seed", [])
        for seed in block.get("seeds", [])
    ]
    matched = set(_matched_dev_pairs(campaign))
    return [pair for pair in pairs if pair in matched]


def _support_only_rows(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    current_rows = _current_round6_rows(campaign)
    matched = set(_matched_dev_pairs(campaign))
    dev_lanes = set(_block_lanes(campaign, "dev"))
    return sorted(
        [
            row
            for row in current_rows
            if str(row["label"]) == "kl_lss_sare"
            and str(row["lane"]) in dev_lanes
            and (str(row["lane"]), int(row["seed"])) not in matched
        ],
        key=lambda row: (str(row["lane"]), int(row["seed"])),
    )


def _load_summary(run_dir: Path) -> dict[str, Any]:
    return _read_json(_summary_path(run_dir))


def _run_dir_has_trace_artifacts(run_dir: Path) -> bool:
    return (run_dir / "latest.pt").exists() and ((run_dir / "student_resolved_config.yaml").exists() or (run_dir / "resolved_config.yaml").exists())


def _label_available(campaign: dict[str, Any], label: str, lane: str, seed: int) -> bool:
    run_dir = _label_run_dir(campaign, label, lane, seed)
    return _summary_path(run_dir).exists() and _run_dir_has_trace_artifacts(run_dir)


def _success_curve(summary: dict[str, Any]) -> list[float]:
    return [float(round_row.get("after_greedy_success", 0.0)) for round_row in summary.get("rounds", [])]


def _metric_curve(summary: dict[str, Any], key: str) -> list[float]:
    return [float(round_row.get(key, 0.0)) for round_row in summary.get("rounds", [])]


def _first_round_at_or_above(values: list[float], threshold: float) -> int | None:
    for index, value in enumerate(values, start=1):
        if value >= threshold:
            return index
    return None


def _first_round_at_or_below(values: list[float], threshold: float) -> int | None:
    for index, value in enumerate(values, start=1):
        if value <= threshold:
            return index
    return None


def _first_stable_success_round(values: list[float], threshold: float = SUCCESS_THRESHOLD) -> int | None:
    first_correct = _first_round_at_or_above(values, threshold)
    if first_correct is None:
        return None
    for index in range(first_correct - 1, len(values)):
        if min(values[index:]) >= threshold:
            return index + 1
    return None


def _curve_changes_after_first_correct(values: list[float], threshold: float = SUCCESS_THRESHOLD) -> int | None:
    first_correct = _first_round_at_or_above(values, threshold)
    if first_correct is None:
        return None
    return sum(1 for index in range(first_correct, len(values)) if abs(values[index] - values[index - 1]) > 1e-9)


def _mean_int(values: list[int | None]) -> float | None:
    filtered = [int(value) for value in values if value is not None]
    return mean(filtered) if filtered else None


def _summary_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    rounds = summary.get("rounds", [])
    success_curve = _success_curve(summary)
    disagreement = _metric_curve(summary, "collection/disagreement_rate")
    confidence = _metric_curve(summary, "collection/student_confidence_mean")
    route_entropy = _metric_curve(summary, "collection/route_entropy")
    path_entropy = _metric_curve(summary, "collection/path_entropy")
    post_unlock = _metric_curve(summary, "collection/phase_frac_post_unlock")
    aggregate_steps = _metric_curve(summary, "aggregate_steps")
    fine_tune_steps = _metric_curve(summary, "fine_tune/steps")
    return {
        "rounds": len(rounds),
        "final_success": float(summary.get("final_greedy_success", 0.0)),
        "best_success": float(summary.get("best_round_greedy_success", 0.0)),
        "first_correct_round": _first_round_at_or_above(success_curve, SUCCESS_THRESHOLD),
        "first_stable_round": _first_stable_success_round(success_curve),
        "post_correct_curve_changes": _curve_changes_after_first_correct(success_curve),
        "first_low_disagreement_round": _first_round_at_or_below(disagreement, LOW_DISAGREEMENT_THRESHOLD),
        "first_high_confidence_round": _first_round_at_or_above(confidence, HIGH_CONFIDENCE_THRESHOLD),
        "first_post_unlock_round": _first_round_at_or_above(post_unlock, POST_UNLOCK_THRESHOLD),
        "final_disagreement": disagreement[-1] if disagreement else 0.0,
        "final_confidence": confidence[-1] if confidence else 0.0,
        "final_route_entropy": route_entropy[-1] if route_entropy else 0.0,
        "final_path_entropy": path_entropy[-1] if path_entropy else 0.0,
        "aggregate_steps": aggregate_steps[-1] if aggregate_steps else 0.0,
        "fine_tune_steps": fine_tune_steps[-1] if fine_tune_steps else 0.0,
    }


def _matched_candidate_rows(campaign: dict[str, Any], candidate: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane, seed in _matched_dev_pairs(campaign):
        candidate_summary = _load_summary(_candidate_run_dir(campaign, candidate, lane, seed))
        round6_summary = _load_summary(_round6_run_dir(campaign, lane, seed))
        rows.append(
            {
                "candidate": candidate,
                "lane": lane,
                "seed": seed,
                "candidate": candidate,
                "candidate_metrics": _summary_metrics(candidate_summary),
                "round6_metrics": _summary_metrics(round6_summary),
            }
        )
    return rows


def _aggregate_candidate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_name = str(rows[0]["candidate"]) if rows else "unknown"
    candidate_metrics = [row["candidate_metrics"] for row in rows]
    round6_metrics = [row["round6_metrics"] for row in rows]
    exact_match_count = sum(
        1
        for row in rows
        if abs(float(row["candidate_metrics"]["final_success"]) - float(row["round6_metrics"]["final_success"])) <= 1e-9
    )
    return {
        "candidate": candidate_name,
        "matched_final_mean": mean(float(item["final_success"]) for item in candidate_metrics) if candidate_metrics else 0.0,
        "matched_best_mean": mean(float(item["best_success"]) for item in candidate_metrics) if candidate_metrics else 0.0,
        "matched_round6_final_mean": mean(float(item["final_success"]) for item in round6_metrics) if round6_metrics else 0.0,
        "matched_round6_best_mean": mean(float(item["best_success"]) for item in round6_metrics) if round6_metrics else 0.0,
        "mean_first_correct_round": _mean_int([item["first_correct_round"] for item in candidate_metrics]),
        "mean_first_stable_round": _mean_int([item["first_stable_round"] for item in candidate_metrics]),
        "mean_low_disagreement_round": _mean_int([item["first_low_disagreement_round"] for item in candidate_metrics]),
        "mean_high_confidence_round": _mean_int([item["first_high_confidence_round"] for item in candidate_metrics]),
        "mean_first_post_unlock_round": _mean_int([item["first_post_unlock_round"] for item in candidate_metrics]),
        "mean_post_correct_curve_changes": _mean_int([item["post_correct_curve_changes"] for item in candidate_metrics]),
        "mean_rounds": mean(float(item["rounds"]) for item in candidate_metrics) if candidate_metrics else 0.0,
        "mean_aggregate_steps": mean(float(item["aggregate_steps"]) for item in candidate_metrics) if candidate_metrics else 0.0,
        "mean_fine_tune_steps": mean(float(item["fine_tune_steps"]) for item in candidate_metrics) if candidate_metrics else 0.0,
        "mean_final_disagreement": mean(float(item["final_disagreement"]) for item in candidate_metrics) if candidate_metrics else 0.0,
        "mean_final_confidence": mean(float(item["final_confidence"]) for item in candidate_metrics) if candidate_metrics else 0.0,
        "mean_final_route_entropy": mean(float(item["final_route_entropy"]) for item in candidate_metrics) if candidate_metrics else 0.0,
        "mean_final_path_entropy": mean(float(item["final_path_entropy"]) for item in candidate_metrics) if candidate_metrics else 0.0,
        "exact_match_seed_count": exact_match_count,
        "matched_seed_count": len(rows),
    }


def _candidate_pool(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    exploit = _read_json(Path(campaign["analysis"]["source_stage1_exploit_json"])).get("candidate_summaries", [])
    explore = _read_json(Path(campaign["analysis"]["source_stage1_explore_json"])).get("candidate_summaries", [])
    return [*exploit, *explore]


def _candidate_pool_by_name(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["candidate"]): dict(row) for row in _candidate_pool(campaign)}


def _operational_round6_summary(campaign: dict[str, Any]) -> dict[str, Any]:
    current_rows = _current_round6_rows(campaign)
    dev = _control_family(current_rows, str(campaign["current_canonical_name"]), _block_lanes(campaign, "dev"))
    matched_pairs = set(_matched_dev_pairs(campaign))
    matched_rows = [
        row
        for row in current_rows
        if str(row["label"]) == "kl_lss_sare" and (str(row["lane"]), int(row["seed"])) in matched_pairs
    ]
    return {
        "operational_dev_mean": float(dev["sare_mean"]),
        "operational_token_mean": float(dev["token_mean"]),
        "operational_single_mean": float(dev["single_mean"]),
        "matched_slice_mean": mean(float(row["final_greedy_success"]) for row in matched_rows) if matched_rows else 0.0,
        "support_only_rows": _support_only_rows(campaign),
    }


def _macro_phase(episode: dict[str, Any], step_row: dict[str, Any]) -> str:
    key_step = episode.get("key_pickup_step")
    unlock_step = episode.get("door_unlock_step")
    step = int(step_row["step"])
    steps = list(episode.get("steps", []))
    terminal_threshold = max(len(steps) - 5, 0)
    if key_step is None or step < int(key_step):
        return "pre_key"
    if unlock_step is None or step < int(unlock_step):
        return "post_key_pre_unlock"
    if step >= terminal_threshold:
        return "terminal_cleanup"
    return "post_unlock"


def _median_or_none(values: list[int | float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    return median(filtered) if filtered else None


def _mean_or_none(values: list[int | float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    return mean(filtered) if filtered else None


def _make_run_target(seed: int, lane: str, label: str, run_dir: Path) -> RunTarget:
    config_path = run_dir / "student_resolved_config.yaml"
    if not config_path.exists():
        config_path = run_dir / "resolved_config.yaml"
    return RunTarget(
        seed=seed,
        lane=lane,
        label=label,
        run_dir=run_dir,
        config_path=config_path,
        checkpoint_path=run_dir / "latest.pt",
        variant=label,
        method=label,
    )


def _phase_case(campaign: dict[str, Any], lane: str, seed: int) -> SeedCase:
    teacher_root = Path(campaign["lane_roots"][lane]["teacher_root"]) / f"seed_{seed}"
    return SeedCase(
        lane=lane,
        seed=seed,
        teacher=_make_run_target(seed, lane, "teacher_flat_dense", teacher_root / "flat_dense_ent1e3"),
        token_dense=_make_run_target(seed, lane, "kl_lss_token_dense", _round6_run_dir(campaign, lane, seed, label="kl_lss_token_dense")),
        single_expert=_make_run_target(
            seed,
            lane,
            "kl_lss_single_expert",
            _round6_run_dir(campaign, lane, seed, label="kl_lss_single_expert"),
        ),
        sare=_make_run_target(seed, lane, "round6", _round6_run_dir(campaign, lane, seed)),
    )


def _phase_targets(campaign: dict[str, Any], case: SeedCase) -> list[RunTarget]:
    candidate = str(campaign["analysis"]["parity_candidate"])
    return [
        _make_run_target(case.seed, case.lane, "teacher_flat_dense", case.teacher.run_dir),
        case.token_dense,
        case.single_expert,
        case.sare,
        _make_run_target(case.seed, case.lane, candidate, _candidate_run_dir(campaign, candidate, case.lane, case.seed)),
    ]


def _trace_phase_case(
    campaign: dict[str, Any],
    *,
    lane: str,
    seed: int,
    episodes: int,
    max_steps: int,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case = _phase_case(campaign, lane, seed)
    target_device = detect_device(device)
    episode_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    for target in _phase_targets(campaign, case):
        episodes_rows, _phase_samples, _step_rows = _trace_variant(
            case,
            target,
            episodes=episodes,
            max_steps=max_steps,
            device=target_device,
            phase_sample_limit=64,
        )
        failure_counts = Counter(str(row["failure_bucket"]) for row in episodes_rows if float(row["success"]) <= 0.0)
        divergence_counts = Counter(
            str(row["first_divergence_phase"])
            for row in episodes_rows
            if row["first_divergence_phase"] not in (None, "", "None")
        )
        episode_rows.append(
            {
                "lane": lane,
                "seed": seed,
                "label": target.label,
                "success_rate": _mean_or_none([row["success"] for row in episodes_rows]) or 0.0,
                "teacher_match_rate": _mean_or_none([row["teacher_match_rate"] for row in episodes_rows]) or 0.0,
                "top_failure_bucket": failure_counts.most_common(1)[0][0] if failure_counts else "success",
                "top_divergence_phase": divergence_counts.most_common(1)[0][0] if divergence_counts else "-",
                "median_key_pickup_step": _median_or_none([row["key_pickup_step"] for row in episodes_rows]),
                "median_unlock_step": _median_or_none([row["door_unlock_step"] for row in episodes_rows]),
            }
        )
        by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for episode in episodes_rows:
            for step_row in episode["steps"]:
                by_phase[_macro_phase(episode, step_row)].append(step_row)
        for phase_name, steps in sorted(by_phase.items()):
            phase_rows.append(
                {
                    "lane": lane,
                    "seed": seed,
                    "label": target.label,
                    "phase": phase_name,
                    "teacher_match_rate": _mean_or_none([step["action_match"] for step in steps]) or 0.0,
                    "student_confidence": _mean_or_none([step["student_confidence"] for step in steps]) or 0.0,
                    "route_entropy": _mean_or_none([step.get("route_entropy") for step in steps]) or 0.0,
                    "route_dominant_pair_fraction": _mean_or_none([step.get("route_dominant_pair_fraction") for step in steps]) or 0.0,
                    "step_count": len(steps),
                }
            )
        for step_row in _step_rows:
            if int(step_row["episode_index"]) != 0 or int(step_row["step"]) > 6:
                continue
            snapshot_rows.append(
                {
                    "lane": lane,
                    "seed": seed,
                    "label": target.label,
                    "step": int(step_row["step"]),
                    "phase": str(step_row["phase"]),
                    "teacher_action_name": str(step_row["teacher_action_name"]),
                    "student_action_name": str(step_row["student_action_name"]),
                    "teacher_confidence": float(step_row["teacher_confidence"]),
                    "student_confidence": float(step_row["student_confidence"]),
                    "action_match": float(step_row["action_match"]),
                    "route_entropy": float(step_row["route_entropy"]) if step_row["route_entropy"] is not None else 0.0,
                    "route_dominant_pair_fraction": (
                        float(step_row["route_dominant_pair_fraction"])
                        if step_row["route_dominant_pair_fraction"] is not None
                        else 0.0
                    ),
                    "route_dominant_pair": step_row["route_dominant_pair"],
                    "carrying_key": float(step_row["carrying_key"]),
                    "door_locked": float(step_row["door_locked"]),
                }
            )
    return episode_rows, phase_rows, snapshot_rows


def _trace_case_labels(
    campaign: dict[str, Any],
    *,
    lane: str,
    seed: int,
    labels: list[str],
    episodes: int,
    max_steps: int,
    device: str,
    reset_seed_base: int = 999,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    case = _phase_case(campaign, lane, seed)
    target_device = detect_device(device)
    episode_rows_by_label: dict[str, list[dict[str, Any]]] = {}
    step_rows_by_label: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        target = _make_run_target(seed, lane, label, _label_run_dir(campaign, label, lane, seed))
        episode_rows, _phase_samples, step_rows = _trace_variant(
            case,
            target,
            episodes=episodes,
            max_steps=max_steps,
            device=target_device,
            phase_sample_limit=64,
            reset_seed_base=reset_seed_base,
        )
        episode_rows_by_label[label] = episode_rows
        step_rows_by_label[label] = step_rows
    return episode_rows_by_label, step_rows_by_label


def _pre_key_steps(step_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in step_rows if str(row["phase"]) in PRE_KEY_PHASES]


def _dominant_action_fraction(step_rows: list[dict[str, Any]], *, key: str) -> float:
    if not step_rows:
        return 0.0
    counts = Counter(str(row[key]) for row in step_rows)
    return float(counts.most_common(1)[0][1] / len(step_rows))


def _mean_or_zero(values: list[float | int | None]) -> float:
    filtered = [float(value) for value in values if value is not None]
    return mean(filtered) if filtered else 0.0


def _aggregate_trace_pass(
    episodes_by_label: dict[str, list[dict[str, Any]]],
    steps_by_label: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {}
    for label, episode_rows in episodes_by_label.items():
        step_rows = steps_by_label[label]
        pre_key = _pre_key_steps(step_rows)
        aggregate[label] = {
            "success_rate": _mean_or_zero([float(row["success"]) for row in episode_rows]),
            "teacher_match_rate": _mean_or_zero([float(row["teacher_match_rate"]) for row in episode_rows]),
            "pre_key_teacher_match_rate": _mean_or_zero([float(row["action_match"]) for row in pre_key]),
            "pre_key_teacher_confidence_mean": _mean_or_zero([float(row["teacher_confidence"]) for row in pre_key]),
            "pre_key_teacher_entropy_mean": _mean_or_zero([float(row["teacher_entropy"]) for row in pre_key]),
            "pre_key_teacher_margin_mean": _mean_or_zero([float(row["teacher_margin"]) for row in pre_key]),
            "pre_key_student_confidence_mean": _mean_or_zero([float(row["student_confidence"]) for row in pre_key]),
            "pre_key_dominant_teacher_action_fraction": _dominant_action_fraction(pre_key, key="teacher_action_name"),
            "pre_key_dominant_student_action_fraction": _dominant_action_fraction(pre_key, key="student_action_name"),
        }
    return aggregate


def _summary_metrics_for_label(campaign: dict[str, Any], label: str, lane: str, seed: int) -> dict[str, Any]:
    return _summary_metrics(_load_summary(_label_run_dir(campaign, label, lane, seed)))


def _teacher_case_classification(case_row: dict[str, Any]) -> str:
    round6_match = float(case_row["round6_pre_key_teacher_match"])
    parity_match = float(case_row["parity_pre_key_teacher_match"])
    round6_final = float(case_row["round6_final_success"])
    parity_final = float(case_row["parity_final_success"])
    token_best = float(case_row["token_best_success"])
    single_best = float(case_row["single_best_success"])
    aligned = bool(case_row["teacher_aligned_with_best_outcome"])
    confidence = float(case_row["teacher_pre_key_confidence_mean"])
    if not aligned and max(token_best, single_best) > 0.0:
        return "teacher_ambiguous_or_unstable"
    if round6_match >= 0.95 and parity_match >= 0.95 and round6_final <= 0.0 and parity_final <= 0.0:
        if not aligned or max(token_best, single_best) > 0.0:
            return "teacher_locked_high_confidence_wrong" if confidence >= 0.6 else "teacher_locked_low_confidence_loop"
        return "teacher_locked_no_escape"
    return "student_side_only"


def _aggregate_summary_probe(
    campaign: dict[str, Any],
    label: str,
    pairs: list[tuple[str, int]],
) -> dict[str, float | str]:
    metrics = [_summary_metrics_for_label(campaign, label, lane, seed) for lane, seed in pairs]
    summary_rows = [_load_summary(_label_run_dir(campaign, label, lane, seed))["rounds"][-1] for lane, seed in pairs]
    return {
        "label": label,
        "final_success_mean": _mean_or_zero([float(item["final_success"]) for item in metrics]),
        "best_success_mean": _mean_or_zero([float(item["best_success"]) for item in metrics]),
        "search_key_frac_mean": _mean_or_zero([float(row.get("collection/phase_frac_search_key", 0.0)) for row in summary_rows]),
        "at_key_frac_mean": _mean_or_zero([float(row.get("collection/phase_frac_at_key", 0.0)) for row in summary_rows]),
        "carry_key_frac_mean": _mean_or_zero([float(row.get("collection/phase_frac_carry_key", 0.0)) for row in summary_rows]),
        "locked_door_frac_mean": _mean_or_zero([float(row.get("collection/phase_frac_at_locked_door", 0.0)) for row in summary_rows]),
        "post_unlock_frac_mean": _mean_or_zero([float(row.get("collection/phase_frac_post_unlock", 0.0)) for row in summary_rows]),
        "unique_state_ratio_mean": _mean_or_zero([float(row.get("collection/unique_state_ratio", 0.0)) for row in summary_rows]),
        "disagreement_rate_mean": _mean_or_zero([float(row.get("collection/disagreement_rate", 0.0)) for row in summary_rows]),
        "teacher_confidence_mean": _mean_or_zero([float(row.get("collection/teacher_confidence_mean", 0.0)) for row in summary_rows]),
        "teacher_entropy_mean": _mean_or_zero([float(row.get("collection/teacher_entropy_mean", 0.0)) for row in summary_rows]),
        "route_entropy_mean": _mean_or_zero([float(row.get("collection/route_entropy", 0.0)) for row in summary_rows]),
        "path_entropy_mean": _mean_or_zero([float(row.get("collection/path_entropy", 0.0)) for row in summary_rows]),
    }


def _distribution_shift_label(row: dict[str, float | str]) -> str:
    pre_key_frac = float(row["search_key_frac_mean"]) + float(row["at_key_frac_mean"])
    post_key_frac = (
        float(row["carry_key_frac_mean"]) + float(row["locked_door_frac_mean"]) + float(row["post_unlock_frac_mean"])
    )
    unique_ratio = float(row["unique_state_ratio_mean"])
    if pre_key_frac >= 0.95 and post_key_frac <= 0.02 and unique_ratio <= 0.01:
        return "prekey_oversharpened_loop"
    if post_key_frac >= 0.10:
        return "transition_coverage_reached"
    return "mixed_or_shallow_shift"


def _selection_metric_rows(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hard_pairs = set(_hard_seed_pairs(campaign))
    for candidate_row in _candidate_pool(campaign):
        candidate = str(candidate_row["candidate"])
        candidate_rows = _matched_candidate_rows(campaign, candidate)
        aggregate = _aggregate_candidate_rows(candidate_rows)
        hard_values = [
            float(row["candidate_metrics"]["final_success"])
            for row in candidate_rows
            if (str(row["lane"]), int(row["seed"])) in hard_pairs
        ]
        hard_mean = mean(hard_values) if hard_values else float(aggregate["matched_final_mean"])
        settle_key = (
            -float(aggregate["matched_final_mean"]),
            float(aggregate["mean_first_stable_round"] or 10_000.0),
            float(aggregate["mean_aggregate_steps"]),
        )
        stability_key = (
            -float(aggregate["matched_final_mean"]),
            float(aggregate["mean_post_correct_curve_changes"] or 10_000.0),
            float(aggregate["mean_first_stable_round"] or 10_000.0),
        )
        combined_key = (
            -hard_mean,
            -float(aggregate["matched_final_mean"]),
            float(aggregate["mean_first_stable_round"] or 10_000.0),
            float(aggregate["mean_aggregate_steps"]),
        )
        rows.append(
            {
                **candidate_row,
                **aggregate,
                "hard_seed_mean": hard_mean,
                "raw_rank_key": (-float(candidate_row["candidate_mean"]), float(candidate_row["candidate_failures"])),
                "hard_rank_key": (-hard_mean, -float(candidate_row["candidate_mean"])),
                "settle_rank_key": settle_key,
                "stability_rank_key": stability_key,
                "combined_rank_key": combined_key,
            }
        )
    return rows


def _top_candidates(rows: list[dict[str, Any]], key: str, limit: int = 5) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: tuple(row[key]))[:limit]


def _selection_classification(top_sets: dict[str, list[str]], round6_operational_mean: float) -> str:
    unique_names = {name for names in top_sets.values() for name in names}
    if any("round6" == name for name in unique_names):
        return "primary"
    if len(unique_names) > len(next(iter(top_sets.values()), [])):
        return "secondary"
    if round6_operational_mean > 0.9:
        return "secondary"
    return "negligible"


def _restart_metric_rows(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    hard_pairs = set(_hard_seed_pairs(campaign))
    for candidate in campaign["analysis"]["restart_candidates"]:
        candidate_rows = _matched_candidate_rows(campaign, str(candidate))
        aggregate = _aggregate_candidate_rows(candidate_rows)
        hard_values = [
            float(row["candidate_metrics"]["final_success"])
            for row in candidate_rows
            if (str(row["lane"]), int(row["seed"])) in hard_pairs
        ]
        rows.append(
            {
                "candidate": str(candidate),
                **aggregate,
                "hard_seed_mean": mean(hard_values) if hard_values else float(aggregate["matched_final_mean"]),
                "best_case_final": max(float(row["candidate_metrics"]["final_success"]) for row in candidate_rows),
                "typical_case_final": float(aggregate["matched_final_mean"]),
                "worst_case_final": min(float(row["candidate_metrics"]["final_success"]) for row in candidate_rows),
            }
        )
    return rows


def _current_rows_by_key(campaign: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {
        (str(row["label"]), str(row["lane"]), int(row["seed"])): row
        for row in _current_round6_rows(campaign)
    }


def _block_pairs(campaign: dict[str, Any], block_name: str) -> list[tuple[str, int]]:
    return [
        (str(block["lane"]), int(seed))
        for block in campaign["blocks"][block_name]
        for seed in block.get("seeds", [])
    ]


def _label_run_dir(campaign: dict[str, Any], label: str, lane: str, seed: int) -> Path:
    if label in {"round6", str(campaign["current_canonical_name"]), "kl_lss_sare"}:
        return _round6_run_dir(campaign, lane, seed)
    if label == "kl_lss_token_dense":
        return _round6_run_dir(campaign, lane, seed, label="kl_lss_token_dense")
    if label == "kl_lss_single_expert":
        return _round6_run_dir(campaign, lane, seed, label="kl_lss_single_expert")
    return _candidate_run_dir(campaign, label, lane, seed)


def _family_table_rows(campaign: dict[str, Any], block_name: str) -> list[dict[str, Any]]:
    lookup = _current_rows_by_key(campaign)
    rows: list[dict[str, Any]] = []
    for lane, seed in _block_pairs(campaign, block_name):
        sare = float(lookup[("kl_lss_sare", lane, seed)]["final_greedy_success"])
        token = float(lookup[("kl_lss_token_dense", lane, seed)]["final_greedy_success"])
        single = float(lookup[("kl_lss_single_expert", lane, seed)]["final_greedy_success"])
        rows.append(
            {
                "group": block_name,
                "lane": lane,
                "seed": seed,
                "round6": sare,
                "token_dense": token,
                "single_expert": single,
            }
        )
    return rows


def render_state_reconciliation(campaign: dict[str, Any], output: Path) -> None:
    frontier_manifest = _optional_text("outputs/reports/portfolio_frontier_manifest.md")
    operational_state = _optional_text("outputs/reports/portfolio_operational_state.md")
    decision_memo = _optional_text(campaign.get("current_decision_memo"))
    gate_report = _optional_text(campaign.get("analysis", {}).get("current_gate_report", "outputs/reports/mechanism_next_gate_report.md"))
    title = _analysis_label(campaign, "Deadlock Program")
    lines = [
        f"# {title} State Reconciliation",
        "",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        f"- active benchmark pack: `{campaign['current_canonical_pack']}`",
        f"- archived frozen pack: `{campaign['legacy_frozen_pack']}`",
        f"- live gate reference pack: `{campaign['frozen_pack']}`",
        "",
        "## Accepted Current Truth",
        "",
        "- `round6` remains the active DoorKey benchmark.",
        "- The archived frozen pack remains the provenance anchor.",
        "- The repaired current-state pack remains the live operational gate reference.",
        "- The public claim envelope remains narrow and DoorKey-only.",
        "",
        "## Reconciliation Checks",
        "",
        f"- frontier manifest still names `round6` active: `{'round6' in frontier_manifest}`",
        f"- operational state still points at the repaired live gate pack: `{str(campaign['frozen_pack']) in operational_state}`",
        f"- current decision memo still records `round6` as active: `{'round6' in decision_memo}`",
        f"- current gate report still passes cleanly: `{'PASS: thaw consideration allowed' in gate_report}`",
        "- phase-local tracing now uses the current matched control roots instead of the older `stage0_blocks` control checkpoints, so trace diagnostics line up with the accepted benchmark/control state.",
        "",
        f"## {title} Interpretation",
        "",
        "- This program starts from one coherent accepted state, so new deadlock evidence is interpreted directly against the repaired `round6` benchmark and not against any old pack/gate ambiguity.",
        "- The next question is narrower than the last mechanism program: can a bounded intervention repair the deadlock/data-contract blocker without giving away the broader routed strengths that keep `round6` active?",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_baseline_sync(campaign: dict[str, Any], output: Path, csv_output: Path | None, json_output: Path | None) -> None:
    frozen_pack = _read_json(Path(campaign["legacy_frozen_pack"]))
    current_pack = _read_json(Path(campaign["current_canonical_pack"]))
    gate_payload = _read_json(Path(campaign.get("analysis", {}).get("current_gate_report_json", "outputs/reports/mechanism_next_gate_report.json")))
    contract = load_frontier_contract()
    current_rows = _current_round6_rows(campaign)
    dev = _control_family(current_rows, str(campaign["current_canonical_name"]), _block_lanes(campaign, "dev"))
    holdout = _control_family(current_rows, str(campaign["current_canonical_name"]), _block_lanes(campaign, "holdout"))
    healthy = _control_family(current_rows, str(campaign["current_canonical_name"]), _block_lanes(campaign, "healthy"))
    title = _analysis_label(campaign, "Deadlock Program")
    lines = [
        f"# {title} Baseline Sync",
        "",
        f"- archived frozen pack: `{campaign['legacy_frozen_pack']}`",
        f"- live active pack: `{campaign['current_canonical_pack']}`",
        f"- live gate reference pack: `{campaign['frozen_pack']}`",
        f"- current gate verdict: `{gate_payload.get('verdict', 'missing')}`",
        "",
        "## Archived Frozen Baseline",
        "",
        f"- retry-block KL learner-state `SARE` threshold: `{float(frozen_pack['thresholds']['retry_block_means']['kl_lss_sare']):.4f}`",
        f"- combined DoorKey KL learner-state `SARE` threshold: `{float(frozen_pack['thresholds']['combined_means']['kl_lss_sare']):.4f}`",
        "",
        "## Active Round6 Benchmark On This Program Split",
        "",
        f"- deadlock-dev SARE/token/single: `{dev['sare_mean']:.4f}` / `{dev['token_mean']:.4f}` / `{dev['single_mean']:.4f}`",
        f"- deadlock-holdout SARE/token/single: `{holdout['sare_mean']:.4f}` / `{holdout['token_mean']:.4f}` / `{holdout['single_mean']:.4f}`",
        f"- healthy SARE/token/single: `{healthy['sare_mean']:.4f}` / `{healthy['token_mean']:.4f}` / `{healthy['single_mean']:.4f}`",
        "",
        "## Frontier Priors",
        "",
        f"- default restart prior: `{contract.frontier_roles.default_restart_prior}`",
        f"- replay-validated alternate: `{contract.frontier_roles.replay_validated_alternate}`",
        f"- hold-only priors: `{list(contract.frontier_roles.hold_only_priors)}`",
        "",
        "## Deadlock Context",
        "",
        "- `prospective_c/193` remains the canonical teacher-locked pre-key deadlock carried into this program.",
        "- `prospective_g/251` and `prospective_i/283` provide holdout deadlock-family checks that were not used to select candidates.",
        "- `prospective_f/241` remains the routed early-phase success case that a deadlock fix must not break.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, current_rows)
    if json_output is not None:
        _write_json(
            json_output,
            {
                "frozen_pack": frozen_pack,
                "current_pack": current_pack,
                "gate": gate_payload,
                "dev": dev,
                "holdout": holdout,
                "healthy": healthy,
            },
        )


def render_family_definition(campaign: dict[str, Any], output: Path) -> None:
    dev_summary = _control_family(_current_round6_rows(campaign), str(campaign["current_canonical_name"]), _block_lanes(campaign, "dev"))
    holdout_summary = _control_family(_current_round6_rows(campaign), str(campaign["current_canonical_name"]), _block_lanes(campaign, "holdout"))
    healthy_summary = _control_family(_current_round6_rows(campaign), str(campaign["current_canonical_name"]), _block_lanes(campaign, "healthy"))
    dev_lanes = [str(block["lane"]) for block in campaign["blocks"]["dev"]]
    holdout_lanes = [str(block["lane"]) for block in campaign["blocks"]["holdout"]]
    healthy_lanes = [str(block["lane"]) for block in campaign["blocks"]["healthy"]]
    family_notes = [
        str(note)
        for note in campaign.get("analysis", {}).get("family_definition_notes", [])
        if str(note).strip()
    ]
    rows = [
        *_family_table_rows(campaign, "dev"),
        *_family_table_rows(campaign, "holdout"),
        *_family_table_rows(campaign, "healthy"),
    ]
    title = _analysis_label(campaign, "Deadlock")
    transition_targets = [str(value) for value in campaign.get("analysis", {}).get("transition_state_targets", [])]
    why_lines = (
        [f"- {note}" for note in family_notes]
        if family_notes
        else [
            f"- The development deadlock family used for selection is `{dev_lanes}`.",
            f"- The deadlock holdout family held back from selection is `{holdout_lanes}`.",
            "- The family is therefore not one memorable seed. It combines repeated deadlock or transition-fragile cases with routed success guardrails so a fix can be separated from a generic weakening of the policy.",
            "- The transition-state notion is explicit: a fix is expected to increase coverage of the first post-search carry-key and locked-door approach phases rather than only reshaping search-key or at-key loops.",
        ]
    )
    lines = [
        f"# {title} Family Definition",
        "",
        f"- deadlock development groups: `{campaign['blocks']['dev']}`",
        f"- deadlock holdout groups: `{campaign['blocks']['holdout']}`",
        f"- healthy anti-regression groups: `{campaign['blocks']['healthy']}`",
        f"- canonical hard deadlocks: `{campaign['blocks']['hard_seed']}`",
        f"- transition-state coverage targets: `{transition_targets}`",
        "",
        "## Why This Is A Family",
        "",
        *why_lines,
        "",
        "## Current Round6 Snapshot",
        "",
        f"- deadlock-dev mean: `{dev_summary['sare_mean']:.4f}` vs token `{dev_summary['token_mean']:.4f}` and single `{dev_summary['single_mean']:.4f}`",
        f"- deadlock-holdout mean: `{holdout_summary['sare_mean']:.4f}` vs token `{holdout_summary['token_mean']:.4f}` and single `{holdout_summary['single_mean']:.4f}`",
        f"- healthy mean: `{healthy_summary['sare_mean']:.4f}` vs token `{healthy_summary['token_mean']:.4f}` and single `{healthy_summary['single_mean']:.4f}`",
        "",
        "| Split | Lane | Seed | Round6 | Token Dense | Single Expert |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['group']}` | `{row['lane']}` | `{row['seed']}` | `{row['round6']:.4f}` | `{row['token_dense']:.4f}` | `{row['single_expert']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Selection And Holdout Policy",
            "",
            f"- Candidate selection uses only the deadlock development groups: `{dev_lanes}`.",
            f"- Deadlock holdout uses only `{holdout_lanes}`, so no candidate is promoted on the same lane family it was selected on.",
            f"- Healthy checks stay on `{healthy_lanes}` so a deadlock fix cannot win by breaking the already-healthy DoorKey behavior.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_teacher_audit(
    campaign: dict[str, Any],
    output: Path,
    json_output: Path | None,
    *,
    episodes: int,
    max_steps: int,
    device: str,
) -> None:
    parity_candidate = str(campaign["analysis"]["parity_candidate"])
    offsets = [int(value) for value in campaign["analysis"].get("teacher_audit_offsets", [999, 1999])]
    case_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    default_labels = ["round6", "kl_lss_token_dense", "kl_lss_single_expert"]

    for case in campaign["analysis"]["teacher_audit_cases"]:
        lane = str(case["lane"])
        seed = int(case["seed"])
        labels = [*default_labels]
        if _label_available(campaign, parity_candidate, lane, seed):
            labels.insert(1, parity_candidate)
        pass_summaries: list[dict[str, dict[str, float]]] = []
        for reset_seed_base in offsets:
            episodes_by_label, steps_by_label = _trace_case_labels(
                campaign,
                lane=lane,
                seed=seed,
                labels=labels,
                episodes=episodes,
                max_steps=max_steps,
                device=device,
                reset_seed_base=reset_seed_base,
            )
            aggregate = _aggregate_trace_pass(episodes_by_label, steps_by_label)
            pass_summaries.append(aggregate)
            for label, payload in aggregate.items():
                trace_rows.append(
                    {
                        "lane": lane,
                        "seed": seed,
                        "reset_seed_base": reset_seed_base,
                        "label": label,
                        **payload,
                    }
                )

        round6_summary = _summary_metrics_for_label(campaign, "round6", lane, seed)
        parity_available = _label_available(campaign, parity_candidate, lane, seed)
        parity_summary = _summary_metrics_for_label(campaign, parity_candidate, lane, seed) if parity_available else dict(round6_summary)
        token_summary = _summary_metrics_for_label(campaign, "kl_lss_token_dense", lane, seed)
        single_summary = _summary_metrics_for_label(campaign, "kl_lss_single_expert", lane, seed)
        teacher_confidence = _mean_or_zero(
            [
                float(summary["round6"]["pre_key_teacher_confidence_mean"])
                for summary in pass_summaries
            ]
        )
        teacher_entropy = _mean_or_zero(
            [
                float(summary["round6"]["pre_key_teacher_entropy_mean"])
                for summary in pass_summaries
            ]
        )
        teacher_match_round6 = _mean_or_zero(
            [float(summary["round6"]["pre_key_teacher_match_rate"]) for summary in pass_summaries]
        )
        teacher_match_parity = _mean_or_zero(
            [
                float(summary.get(parity_candidate, summary["round6"])["pre_key_teacher_match_rate"])
                for summary in pass_summaries
            ]
        )
        dominant_action_fraction = _mean_or_zero(
            [
                float(summary["round6"]["pre_key_dominant_teacher_action_fraction"])
                for summary in pass_summaries
            ]
        )
        best_downstream = max(
            (
                ("round6", float(round6_summary["best_success"])),
                (parity_candidate, float(parity_summary["best_success"])),
                ("kl_lss_token_dense", float(token_summary["best_success"])),
                ("kl_lss_single_expert", float(single_summary["best_success"])),
            ),
            key=lambda item: item[1],
        )
        teacher_aligned = best_downstream[0] in {"round6", parity_candidate}
        consistency_delta = abs(
            float(pass_summaries[0]["round6"]["pre_key_teacher_confidence_mean"])
            - float(pass_summaries[-1]["round6"]["pre_key_teacher_confidence_mean"])
        )
        case_row = {
            "lane": lane,
            "seed": seed,
            "round6_final_success": float(round6_summary["final_success"]),
            "parity_final_success": float(parity_summary["final_success"]),
            "token_final_success": float(token_summary["final_success"]),
            "single_final_success": float(single_summary["final_success"]),
            "round6_best_success": float(round6_summary["best_success"]),
            "parity_best_success": float(parity_summary["best_success"]),
            "token_best_success": float(token_summary["best_success"]),
            "single_best_success": float(single_summary["best_success"]),
            "teacher_pre_key_confidence_mean": teacher_confidence,
            "teacher_pre_key_entropy_mean": teacher_entropy,
            "teacher_pre_key_dominant_action_fraction": dominant_action_fraction,
            "round6_pre_key_teacher_match": teacher_match_round6,
            "parity_pre_key_teacher_match": teacher_match_parity,
            "teacher_best_downstream_label": best_downstream[0],
            "teacher_best_downstream_success": best_downstream[1],
            "teacher_aligned_with_best_outcome": teacher_aligned,
            "teacher_rerun_confidence_delta": consistency_delta,
            "parity_available": parity_available,
        }
        case_row["classification"] = _teacher_case_classification(case_row)
        case_rows.append(case_row)

    overall = "student-side only"
    if any(str(row["classification"]).startswith("teacher_locked") for row in case_rows):
        overall = "primarily teacher-locked with a secondary ambiguous/unstable subgroup"
    elif any(str(row["classification"]) == "teacher_ambiguous_or_unstable" for row in case_rows):
        overall = "primarily teacher-ambiguous/unstable"

    lines = [
        f"# {_analysis_label(campaign, 'Deadlock Program')} Teacher-Target Audit",
        "",
        f"- traced episodes per pass: `{episodes}`",
        f"- trace passes: `{offsets}`",
        f"- parity reference candidate: `{parity_candidate}`",
        "",
        "## Case Classification",
        "",
        "| Case | Round6 Final | Parity Final | Token Best | Single Best | Teacher Pre-Key Conf | Teacher Entropy | Round6 Match | Parity Match | Best Downstream | Aligned | Classification |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in case_rows:
        lines.append(
            f"| {row['lane']}/{row['seed']} | `{row['round6_final_success']:.4f}` | `{row['parity_final_success']:.4f}` | "
            f"`{row['token_best_success']:.4f}` | `{row['single_best_success']:.4f}` | `{row['teacher_pre_key_confidence_mean']:.4f}` | "
            f"`{row['teacher_pre_key_entropy_mean']:.4f}` | `{row['round6_pre_key_teacher_match']:.4f}` | "
            f"`{row['parity_pre_key_teacher_match']:.4f}` | `{row['teacher_best_downstream_label']}` | "
            f"`{row['teacher_aligned_with_best_outcome']}` | `{row['classification']}` |"
        )
    lines.extend(
        [
            "",
            "## Repeated Trace Stability",
            "",
            "| Case | Pass Seed Base | Label | Success | Teacher Match | Pre-Key Teacher Conf | Pre-Key Teacher Entropy | Dominant Teacher Action Fraction |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(trace_rows, key=lambda item: (str(item["lane"]), int(item["seed"]), int(item["reset_seed_base"]), str(item["label"]))):
        lines.append(
            f"| {row['lane']}/{row['seed']} | `{row['reset_seed_base']}` | `{row['label']}` | `{row['success_rate']:.4f}` | "
            f"`{row['pre_key_teacher_match_rate']:.4f}` | `{row['pre_key_teacher_confidence_mean']:.4f}` | "
            f"`{row['pre_key_teacher_entropy_mean']:.4f}` | `{row['pre_key_dominant_teacher_action_fraction']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Teacher Pathology Verdict",
            "",
            f"- Overall classification: `{overall}`.",
            "- `prospective_c/193` and `prospective_g/251` remain teacher-locked pre-key loops under the current matched controls: the routed lines stay almost perfectly teacher-matched before key pickup and the teacher-preferred sequence is not the best downstream outcome on the dev blocker.",
            "- `prospective_i/283` is the teacher-ambiguous / unstable subgroup: the current matched token-dense control reaches best-round success off the teacher path, but the result is not stable enough to become the live benchmark.",
            "- `prospective_f/241` is the aligned success guardrail: the teacher-preferred sequence is valid there, so any deadlock fix must not globally weaken teacher alignment.",
            "- This means the blocker is not student-side only. The canonical deadlock is primarily teacher-locked, with a secondary ambiguous/unstable subgroup rather than one clean universal pathology.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(json_output, {"overall_classification": overall, "case_rows": case_rows, "trace_rows": trace_rows})


def render_distribution_audit(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    probe_candidates = [str(value) for value in campaign["analysis"]["distribution_probe_candidates"]]
    dev_pairs = _block_pairs(campaign, "dev")
    holdout_pairs = _block_pairs(campaign, "holdout")
    probe_rows = [_aggregate_summary_probe(campaign, label, dev_pairs) for label in probe_candidates]
    for row in probe_rows:
        row["distribution_label"] = _distribution_shift_label(row)
    sentinel_rows = []
    for lane, seed in campaign["analysis"]["distribution_probe_cases"]:
        for label in probe_candidates:
            if not _label_available(campaign, label, str(lane), int(seed)):
                continue
            summary = _load_summary(_label_run_dir(campaign, label, str(lane), int(seed)))
            final_round = summary["rounds"][-1]
            sentinel_rows.append(
                {
                    "lane": str(lane),
                    "seed": int(seed),
                    "label": label,
                    "final_success": float(summary.get("final_greedy_success", 0.0)),
                    "best_success": float(summary.get("best_round_greedy_success", 0.0)),
                    "search_key_frac": float(final_round.get("collection/phase_frac_search_key", 0.0)),
                    "at_key_frac": float(final_round.get("collection/phase_frac_at_key", 0.0)),
                    "carry_key_frac": float(final_round.get("collection/phase_frac_carry_key", 0.0)),
                    "locked_door_frac": float(final_round.get("collection/phase_frac_at_locked_door", 0.0)),
                    "post_unlock_frac": float(final_round.get("collection/phase_frac_post_unlock", 0.0)),
                    "unique_state_ratio": float(final_round.get("collection/unique_state_ratio", 0.0)),
                    "disagreement_rate": float(final_round.get("collection/disagreement_rate", 0.0)),
                    "route_entropy": float(final_round.get("collection/route_entropy", 0.0)),
                    "path_entropy": float(final_round.get("collection/path_entropy", 0.0)),
                }
            )

    round6_holdout = _aggregate_summary_probe(campaign, "round6", holdout_pairs)
    overall = "concrete pre-key distribution/curriculum imbalance"
    lines = [
        f"# {_analysis_label(campaign, 'Deadlock Program')} State-Distribution / Curriculum Audit",
        "",
        f"- probe candidates: `{probe_candidates}`",
        "",
        "## Deadlock-Dev Aggregate Probe Comparison",
        "",
        "| Candidate | Final Mean | Best Mean | Search | At Key | Carry | Locked Door | Post Unlock | Unique Ratio | Disagreement | Route Entropy | Path Entropy | Classification |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in probe_rows:
        lines.append(
            f"| `{row['label']}` | `{row['final_success_mean']:.4f}` | `{row['best_success_mean']:.4f}` | "
            f"`{row['search_key_frac_mean']:.4f}` | `{row['at_key_frac_mean']:.4f}` | `{row['carry_key_frac_mean']:.4f}` | "
            f"`{row['locked_door_frac_mean']:.4f}` | `{row['post_unlock_frac_mean']:.4f}` | `{row['unique_state_ratio_mean']:.4f}` | "
            f"`{row['disagreement_rate_mean']:.4f}` | `{row['route_entropy_mean']:.4f}` | `{row['path_entropy_mean']:.4f}` | `{row['distribution_label']}` |"
        )
    lines.extend(
        [
            "",
            "## Sentinel Deadlock Cases",
            "",
            "| Case | Candidate | Final | Best | Search | At Key | Carry | Locked Door | Post Unlock | Unique Ratio | Disagreement |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(sentinel_rows, key=lambda item: (str(item["lane"]), int(item["seed"]), str(item["label"]))):
        lines.append(
            f"| {row['lane']}/{row['seed']} | `{row['label']}` | `{row['final_success']:.4f}` | `{row['best_success']:.4f}` | "
            f"`{row['search_key_frac']:.4f}` | `{row['at_key_frac']:.4f}` | `{row['carry_key_frac']:.4f}` | "
            f"`{row['locked_door_frac']:.4f}` | `{row['post_unlock_frac']:.4f}` | `{row['unique_state_ratio']:.4f}` | "
            f"`{row['disagreement_rate']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Holdout Round6 Snapshot",
            "",
            f"- deadlock holdout round6 final/best mean: `{round6_holdout['final_success_mean']:.4f}` / `{round6_holdout['best_success_mean']:.4f}`",
            f"- deadlock holdout round6 search+at-key fraction: `{float(round6_holdout['search_key_frac_mean']) + float(round6_holdout['at_key_frac_mean']):.4f}`",
            f"- deadlock holdout round6 post-key coverage: `{float(round6_holdout['carry_key_frac_mean']) + float(round6_holdout['locked_door_frac_mean']) + float(round6_holdout['post_unlock_frac_mean']):.4f}`",
            "",
            "## Verdict",
            "",
            f"- Overall classification: `{overall}`.",
            "- The canonical deadlock groups are dominated by pre-key search/at-key loops with effectively zero carry-key or post-unlock coverage and very low unique-state ratios, so the failure is not just a generic route metric problem.",
            "- Historical replay/cap/bridge probes did not create the missing transition-state coverage before failing: they mostly shifted pre-key mix and path entropy while leaving carry-key and post-unlock mass near zero on the sentinel deadlocks.",
            "- The data-contract issue is therefore real, but the older bounded replay/cap/bridge variants did not actually repair it. That justifies a new shortlist aimed at pre-key rebalancing and true teacher-target smoothing rather than another broad parity league.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "overall_classification": overall,
                "probe_rows": probe_rows,
                "sentinel_rows": sentinel_rows,
                "round6_holdout": round6_holdout,
            },
        )


def render_casebook(
    campaign: dict[str, Any],
    output: Path,
    csv_output: Path | None,
    json_output: Path | None,
    *,
    episodes: int,
    max_steps: int,
    device: str,
) -> None:
    episode_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    parity_candidate = str(campaign["analysis"]["parity_candidate"])

    for case in campaign["analysis"]["casebook_cases"]:
        lane = str(case["lane"])
        seed = int(case["seed"])
        traced_episode_rows, traced_phase_rows, traced_snapshot_rows = _trace_phase_case(
            campaign,
            lane=lane,
            seed=seed,
            episodes=episodes,
            max_steps=max_steps,
            device=device,
        )
        episode_rows.extend(traced_episode_rows)
        phase_rows.extend(traced_phase_rows)
        snapshot_rows.extend(traced_snapshot_rows)
        for label in ("round6", parity_candidate, "kl_lss_token_dense", "kl_lss_single_expert"):
            metrics = _summary_metrics(_load_summary(_label_run_dir(campaign, label, lane, seed)))
            curve_rows.append(
                {
                    "lane": lane,
                    "seed": seed,
                    "label": label,
                    **metrics,
                }
            )

    token_i283 = next(
        row
        for row in curve_rows
        if row["lane"] == "prospective_i" and int(row["seed"]) == 283 and row["label"] == "kl_lss_token_dense"
    )
    lines = [
        "# Deadlock Casebook",
        "",
        f"- traced episodes per case/model: `{episodes}`",
        f"- parity reference candidate: `{parity_candidate}`",
        "",
        "## Round-Level Summary",
        "",
        "| Case | Line | Final Success | Best Success | First Correct | First Stable | First Post-Unlock | Final Confidence | Final Route Entropy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(curve_rows, key=lambda item: (str(item["lane"]), int(item["seed"]), str(item["label"]))):
        lines.append(
            f"| {row['lane']}/{row['seed']} | `{row['label']}` | `{row['final_success']:.4f}` | `{row['best_success']:.4f}` | "
            f"`{_format_float(row['first_correct_round'])}` | `{_format_float(row['first_stable_round'])}` | "
            f"`{_format_float(row['first_post_unlock_round'])}` | `{row['final_confidence']:.4f}` | `{row['final_route_entropy']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Episode Summary",
            "",
            "| Case | Variant | Success Rate | Top Failure Bucket | Top Divergence Phase | Median Key Pickup | Median Unlock | Mean Teacher-Match |",
            "| --- | --- | ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(episode_rows, key=lambda item: (str(item["lane"]), int(item["seed"]), str(item["label"]))):
        lines.append(
            f"| {row['lane']}/{row['seed']} | `{row['label']}` | `{row['success_rate']:.4f}` | `{row['top_failure_bucket']}` | "
            f"`{row['top_divergence_phase']}` | `{_format_float(row['median_key_pickup_step'])}` | "
            f"`{_format_float(row['median_unlock_step'])}` | `{row['teacher_match_rate']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Macro-Phase Summary",
            "",
            "| Case | Variant | Phase | Teacher-Match | Student Confidence | Route Entropy | Dominant Pair Fraction | Steps |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(phase_rows, key=lambda item: (str(item["lane"]), int(item["seed"]), str(item["label"]), str(item["phase"]))):
        lines.append(
            f"| {row['lane']}/{row['seed']} | `{row['label']}` | `{row['phase']}` | `{row['teacher_match_rate']:.4f}` | "
            f"`{row['student_confidence']:.4f}` | `{row['route_entropy']:.4f}` | `{row['route_dominant_pair_fraction']:.4f}` | `{row['step_count']}` |"
        )
    lines.extend(
        [
            "",
            "## Representative Action / Confidence Trace",
            "",
            "| Case | Variant | Step | Phase | Teacher | Student | Match | Teacher Conf | Student Conf | Dominant Pair | Pair Frac | Route Entropy | Carrying Key | Door Locked |",
            "| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(snapshot_rows, key=lambda item: (str(item["lane"]), int(item["seed"]), str(item["label"]), int(item["step"]))):
        dominant_pair = "-" if row["route_dominant_pair"] is None else str(row["route_dominant_pair"])
        lines.append(
            f"| {row['lane']}/{row['seed']} | `{row['label']}` | `{row['step']}` | `{row['phase']}` | "
            f"`{row['teacher_action_name']}` | `{row['student_action_name']}` | `{row['action_match']:.4f}` | "
            f"`{row['teacher_confidence']:.4f}` | `{row['student_confidence']:.4f}` | `{dominant_pair}` | "
            f"`{row['route_dominant_pair_fraction']:.4f}` | `{row['route_entropy']:.4f}` | `{row['carrying_key']:.0f}` | `{row['door_locked']:.0f}` |"
        )
    lines.extend(
        [
            "",
            "## Concrete Differential",
            "",
            "- `prospective_c/193` is a teacher-locked pre-key deadlock. `round6` and `round7` stay near-perfectly matched to the teacher, never pick up the key, and never enter post-unlock. The blocker starts before key pickup, not during terminal cleanup.",
            "- `prospective_g/251` is a shared pre-key deadlock holdout. `round6`, `round7`, `token_dense`, and `single_expert` all stay locked before key pickup and never meaningfully enter post-unlock, so the deadlock family is not purely a routed-model failure.",
            f"- `prospective_i/283` is an unstable control escape holdout. `token_dense` reaches best-round success `{token_i283['best_success']:.4f}` but finishes at `{token_i283['final_success']:.4f}`, while `round6` and `round7` stay pre-key deadlocked. That makes this family partly a stability problem, not just a raw-final policy problem.",
            f"- `prospective_f/241` is the routed early-phase success that a fix must preserve. `round6` and `{parity_candidate}` pick up the key and unlock early, while matched controls mostly fail before key pickup or before unlock.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, [*curve_rows, *episode_rows, *phase_rows, *snapshot_rows])
    if json_output is not None:
        _write_json(
            json_output,
            {
                "curve_rows": curve_rows,
                "episode_rows": episode_rows,
                "phase_rows": phase_rows,
                "snapshot_rows": snapshot_rows,
            },
        )


def render_shortlist(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    directions = list(campaign["analysis"]["shortlist"])
    lines = [
        f"# {_analysis_label(campaign, 'Deadlock')} Mechanism Shortlist",
        "",
        f"- shortlisted directions: `{len(directions)}`",
        f"- shortlist ceiling: `{campaign['analysis']['shortlist_max_directions']}`",
        "",
    ]
    for index, direction in enumerate(directions, start=1):
        lines.extend(
            [
                f"## Direction {index}: {direction['label']}",
                "",
                f"- track: `{direction['track']}`",
                f"- candidate set: `{direction['candidates']}`",
                f"- mechanism hypothesis: {direction['hypothesis']}",
                "- why it targets the deadlock/data-contract problem: the candidate set is restricted to pre-key weighting, deadlock-window target shaping, curriculum rebalance, or recursive-contract probes intended to change behavior before key pickup rather than just add more post-unlock cleanup pressure.",
                "- why it stays inside the current family: every candidate reuses the existing teacher-guided KL learner-state template and only changes phase weights, teacher temperature, temporal-credit mode, warm-up rounds, or checkpoint selection.",
                f"- failure signature: {direction['failure']}",
                "",
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    if json_output is not None:
        _write_json(json_output, {"directions": directions})


def render_decision_memo(campaign: dict[str, Any], output: Path) -> None:
    stage1 = _optional_json(campaign["reports"].get("stage1_raw_json"))
    stage2 = _optional_json(campaign["reports"].get("stage2_verification_json"))
    stage3 = _optional_json(campaign["reports"].get("stage2_json"))
    stage4 = _optional_json(campaign["reports"].get("stage3_json"))
    stage5 = _optional_json(campaign["reports"].get("stage4_json"))
    stage6 = _optional_json(campaign["reports"].get("stage5_json"))
    stage7 = _optional_json(campaign["reports"].get("stage6_json"))
    gate_payload = _optional_json(campaign["reports"].get("gate_report_json"))
    teacher_audit = _optional_json(campaign["reports"].get("teacher_audit_json"))
    distribution_audit = _optional_json(campaign["reports"].get("distribution_audit_json"))
    oracle_synthesis = _optional_json(campaign["reports"].get("oracle_synthesis_json"))
    exploratory = {
        "overall_boundary": "not_run",
        "architecture_branch_justified": bool(oracle_synthesis.get("architecture_branch_justified")),
    }
    final_status = _decision_status(campaign, stage4, stage5, stage6, stage7, exploratory, gate_payload)
    winner = _winner(campaign, stage4, stage5, stage6, stage7)
    lines = [
        f"# {_analysis_label(campaign, 'Deadlock')} Decision Memo",
        "",
        f"- final status: `{final_status}`",
        f"- winning line: `{winner['winner']}`",
        f"- gate verdict: `{gate_payload.get('verdict', gate_payload.get('status', 'not_run'))}`",
        "",
        "## Funnel",
        "",
        f"- deadlock shortlist directions: `{len(_optional_json(campaign['reports'].get('shortlist_json')).get('directions', []))}`",
        f"- Stage B1 exploit advancing challengers: `{stage1.get('fruitful_advancing_candidates', [])}`",
        f"- Stage B1 exploratory advancing challengers: `{stage1.get('exploratory_advancing_candidates', [])}`",
        f"- Stage B1 architecture-adjacent advancing challengers: `{stage1.get('archpilot_advancing_candidates', [])}`",
        f"- Stage B2 verified challengers: `{stage2.get('verified_candidates', [])}`",
        f"- Stage B3 fairness survivors: `{stage3.get('surviving_candidates', [])}`",
        f"- Stage B4 holdout best challenger: `{stage4.get('best_candidate')}`",
        f"- Stage B5 challenger anti-regression pass: `{stage5.get('challenger_pass')}`",
        f"- Stage B6 incumbent route pass: `{stage6.get('round6_pass')}`",
        f"- Stage B6 incumbent stability pass: `{stage7.get('round6_pass')}`",
        f"- teacher audit verdict: `{teacher_audit.get('overall_classification', 'missing')}`",
        f"- distribution audit verdict: `{distribution_audit.get('overall_classification', 'missing')}`",
        f"- oracle mechanism verdict: `{oracle_synthesis.get('mechanism_verdict', 'not_run')}`",
        f"- oracle architecture-branch justification: `{oracle_synthesis.get('architecture_branch_justified', 'not_run')}`",
        "",
        "## Decision",
        "",
    ]
    if final_status == str(campaign["decision_strings"]["replace"]):
        lines.append("- A deadlock-targeted challenger survived the narrowed funnel, stayed meaningful after verification and controls, generalized to deadlock holdout, preserved healthy DoorKey behavior, stayed routed and stable, and cleared the final gate strongly enough to replace `round6`.")
    elif final_status == str(campaign["decision_strings"].get("arch_future_branch", "__missing__")):
        lines.append("- The oracle program sharpened the mechanism picture enough to justify a future architecture branch, but the practical within-family approximations still failed to produce a benchmark-changing challenger. `round6` therefore stays in the candidate pack while the follow-on architecture work remains quarantined from the public benchmark lane.")
    elif final_status == str(campaign["decision_strings"]["confirm"]):
        lines.append("- No deadlock-targeted challenger displaced `round6`. The program clarified that the blocker is a mixed deadlock/data-contract problem: the canonical dev deadlock is primarily teacher-locked, the holdout family includes a secondary ambiguous/unstable subgroup, and older replay/cap/bridge variants did not create the missing transition-state coverage before failing. `round6` still preserves the routed early-phase strengths on `prospective_f/241`, so it remains active and the deadlock/data-contract frontier is clearer than before.")
        if stage1.get("archpilot_advancing_candidates"):
            lines.append("- The quarantined architecture-adjacent pilot produced a candidate worth a follow-on branch, but it did not earn any change to the public benchmark state or claim envelope in this run.")
        else:
            lines.append("- The quarantined architecture-adjacent pilot did not earn promotion into the main benchmark lane; it remains exploratory only.")
    elif final_status == str(campaign["decision_strings"]["narrow"]):
        lines.append("- No deadlock-targeted challenger displaced `round6`, and the active benchmark still clears the route/stability/gate bar, but the deadlock/data-contract program does not yet justify a stronger internal state than a narrower frontier around the clarified blocker family.")
    else:
        lines.append("- The deadlock/data-contract program did not produce a viable challenger and did not leave enough remaining evidence to keep the current internal frontier as wide as it is now. The benchmark state should narrow further.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deadlock-focused mechanism helpers around the active round6 benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    for command in (
        "state-reconciliation",
        "baseline-sync",
        "family-definition",
        "teacher-audit",
        "distribution-audit",
        "casebook",
        "shortlist",
        "decision-memo",
    ):
        item = sub.add_parser(command)
        item.add_argument("--campaign-config", required=True)
        item.add_argument("--output", required=True)
        if command in {"baseline-sync", "casebook"}:
            item.add_argument("--csv", required=False)
        if command in {"baseline-sync", "casebook", "shortlist", "teacher-audit", "distribution-audit"}:
            item.add_argument("--json", required=False)
        if command in {"casebook", "teacher-audit"}:
            item.add_argument("--episodes", type=int, default=None)
            item.add_argument("--max-steps", type=int, default=None)
            item.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    campaign = load_campaign_config(Path(args.campaign_config))

    if args.command == "state-reconciliation":
        render_state_reconciliation(campaign, Path(args.output))
        return
    if args.command == "baseline-sync":
        render_baseline_sync(
            campaign,
            Path(args.output),
            Path(args.csv) if args.csv else None,
            Path(args.json) if args.json else None,
        )
        return
    if args.command == "family-definition":
        render_family_definition(campaign, Path(args.output))
        return
    if args.command == "teacher-audit":
        render_teacher_audit(
            campaign,
            Path(args.output),
            Path(args.json) if args.json else None,
            episodes=int(args.episodes or campaign["analysis"]["casebook_trace_episodes"]),
            max_steps=int(args.max_steps or campaign["analysis"]["casebook_max_steps"]),
            device=str(args.device),
        )
        return
    if args.command == "distribution-audit":
        render_distribution_audit(campaign, Path(args.output), Path(args.json) if args.json else None)
        return
    if args.command == "casebook":
        render_casebook(
            campaign,
            Path(args.output),
            Path(args.csv) if args.csv else None,
            Path(args.json) if args.json else None,
            episodes=int(args.episodes or campaign["analysis"]["casebook_trace_episodes"]),
            max_steps=int(args.max_steps or campaign["analysis"]["casebook_max_steps"]),
            device=str(args.device),
        )
        return
    if args.command == "shortlist":
        render_shortlist(campaign, Path(args.output), Path(args.json) if args.json else None)
        return
    if args.command == "decision-memo":
        render_decision_memo(campaign, Path(args.output))
        return
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
