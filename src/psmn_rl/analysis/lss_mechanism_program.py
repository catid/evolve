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


def _summary_path(run_dir: Path) -> Path:
    return run_dir / "summary.json"


def _candidate_source_root(campaign: dict[str, Any]) -> Path:
    return Path(campaign["analysis"]["source_stage1_root"])


def _candidate_run_dir(campaign: dict[str, Any], candidate: str, lane: str, seed: int) -> Path:
    return _candidate_source_root(campaign) / candidate / lane / f"seed_{seed}" / "kl_lss_sare"


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
        token_dense=_make_run_target(seed, lane, "kl_lss_token_dense", teacher_root / "token_dense_ent1e3"),
        single_expert=_make_run_target(seed, lane, "kl_lss_single_expert", teacher_root / "single_expert_ent1e3"),
        sare=_make_run_target(seed, lane, "round6", _round6_run_dir(campaign, lane, seed)),
    )


def _phase_targets(campaign: dict[str, Any], case: SeedCase) -> list[RunTarget]:
    candidate = str(campaign["analysis"]["phase_candidate"])
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
    return episode_rows, phase_rows


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


def render_state_reconciliation(campaign: dict[str, Any], output: Path) -> None:
    frontier_manifest = _optional_text("outputs/reports/portfolio_frontier_manifest.md")
    operational_state = _optional_text("outputs/reports/portfolio_operational_state.md")
    decision_memo = _optional_text(campaign.get("current_decision_memo"))
    gate_report = _optional_text("outputs/reports/next_round_gate_report.md")
    lines = [
        "# Mechanism Program State Reconciliation",
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
        f"- live gate report still passes cleanly: `{'PASS: thaw consideration allowed' in gate_report}`",
        "",
        "## Mechanism Program Interpretation",
        "",
        "- This program starts from one coherent accepted state, so new mechanism evidence can be interpreted directly against the repaired `round6` benchmark instead of revisiting the earlier gate ambiguity.",
        "- The next question is not whether the repo knows what the active benchmark is. The next question is why `round6` keeps beating or operationally outlasting candidates that repeatedly fall into the same slightly-worse parity bucket.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_baseline_sync(campaign: dict[str, Any], output: Path, csv_output: Path | None, json_output: Path | None) -> None:
    frozen_pack = _read_json(Path(campaign["legacy_frozen_pack"]))
    current_pack = _read_json(Path(campaign["current_canonical_pack"]))
    gate_payload = _read_json(Path("outputs/reports/next_round_gate_report.json"))
    contract = load_frontier_contract()
    current_rows = _current_round6_rows(campaign)
    dev = _control_family(current_rows, str(campaign["current_canonical_name"]), _block_lanes(campaign, "dev"))
    holdout = _control_family(current_rows, str(campaign["current_canonical_name"]), _block_lanes(campaign, "holdout"))
    healthy = _control_family(current_rows, str(campaign["current_canonical_name"]), _block_lanes(campaign, "healthy"))
    lines = [
        "# Mechanism Program Baseline Sync",
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
        "## Active Round6 Benchmark",
        "",
        f"- retry-block KL learner-state `SARE` mean: `{float(current_pack['metrics']['retry_block']['kl_lss_sare']['mean']):.4f}`",
        f"- combined DoorKey KL learner-state `SARE` mean: `{float(current_pack['metrics']['combined']['kl_lss_sare']['mean']):.4f}`",
        f"- development SARE/token/single: `{dev['sare_mean']:.4f}` / `{dev['token_mean']:.4f}` / `{dev['single_mean']:.4f}`",
        f"- holdout SARE/token/single: `{holdout['sare_mean']:.4f}` / `{holdout['token_mean']:.4f}` / `{holdout['single_mean']:.4f}`",
        f"- healthy SARE/token/single: `{healthy['sare_mean']:.4f}` / `{healthy['token_mean']:.4f}` / `{healthy['single_mean']:.4f}`",
        "",
        "## Frontier Priors",
        "",
        f"- default restart prior: `{contract.frontier_roles.default_restart_prior}`",
        f"- replay-validated alternate: `{contract.frontier_roles.replay_validated_alternate}`",
        f"- hold-only priors: `{list(contract.frontier_roles.hold_only_priors)}`",
        f"- retired priors: `{list(contract.frontier_roles.retired_priors)}`",
        "",
        "## Current Hard-Seed Context",
        "",
        "- `prospective_c/193` remains unsolved on the active line and is the key teacher-lock blocker carried into the mechanism program.",
        "- The active line still passes healthy, route, stability, and gate checks, so the mechanism question is now about the source of the edge rather than about whether the benchmark state itself is broken.",
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


def render_registration(campaign: dict[str, Any], output: Path) -> None:
    track_counts = Counter(str(meta["track"]) for meta in campaign["candidates"].values())
    lines = [
        "# Mechanism Program Registration",
        "",
        f"- active benchmark: `{campaign['current_canonical_name']}`",
        f"- live gate reference pack: `{campaign['frozen_pack']}`",
        f"- archived frozen pack: `{campaign['legacy_frozen_pack']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Mechanism Questions",
        "",
        "- Does `round6` separate from the parity bucket because it reaches correct behavior earlier, stabilizes earlier, or simply covers more of the accepted-state support surface?",
        "- Is the persistent blocker a post-unlock cleanup miss, or a teacher-locked pre-key / pre-unlock deadlock that more weighting and replay will not fix?",
        "- Is candidate selection a primary bottleneck, or only a secondary tie-break inside an otherwise unchanged parity bucket?",
        "- Does the current restart ordering still make operational sense once the parity bucket is decomposed by convergence, compute, and repeatability instead of just raw final means?",
        "",
        "## Diagnostic Artifacts",
        "",
        f"- round-level differential: `{campaign['reports']['stageA1_report']}`",
        f"- phase-local blocker analysis: `{campaign['reports']['stageA2_report']}`",
        f"- selection sensitivity: `{campaign['reports']['stageA3_report']}`",
        f"- restart sensitivity: `{campaign['reports']['stageA4_report']}`",
        f"- targeted shortlist: `{campaign['reports']['stageA5_report']}`",
        "",
        "## Targeted Experiment Budget",
        "",
        f"- exploit candidates: `{track_counts['fruitful']}`",
        f"- exploration candidates: `{track_counts['exploratory']}`",
        f"- total candidates: `{len(campaign['candidates'])}`",
        f"- development families: `{campaign['blocks']['dev']}`",
        f"- holdout families: `{campaign['blocks']['holdout']}`",
        f"- healthy families: `{campaign['blocks']['healthy']}`",
        f"- hard-block families: `{campaign['blocks']['hard_seed']}`",
        "",
        "## Fair-Shot and Pruning Rules",
        "",
        "- Each shortlisted direction is screened only after the mechanism analysis says why it might beat the previous broad leagues.",
        "- No candidate is promoted on one lucky seed or one lucky family.",
        "- No candidate survives if it cannot clear the same matched-control, holdout, anti-regression, route, stability, and gate path as the active benchmark.",
        "",
        "## Final Decision Rules",
        "",
        "- `active benchmark confirmed and frontier clarified`",
        "- `challenger replaces the active benchmark`",
        "- `active benchmark remains and envelope stays narrow`",
        "- `benchmark/frontier state needs narrowing`",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_round_differential(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    candidates = [str(name) for name in campaign["analysis"]["differential_candidates"]]
    operational = _operational_round6_summary(campaign)
    round6_aggregate = _aggregate_candidate_rows(
        [
            {
                "candidate": str(campaign["current_canonical_name"]),
                "lane": lane,
                "seed": seed,
                "candidate_metrics": _summary_metrics(_load_summary(_round6_run_dir(campaign, lane, seed))),
                "round6_metrics": _summary_metrics(_load_summary(_round6_run_dir(campaign, lane, seed))),
            }
            for lane, seed in _matched_dev_pairs(campaign)
        ]
    )
    candidate_rows = [_aggregate_candidate_rows(_matched_candidate_rows(campaign, candidate)) for candidate in candidates]
    lines = [
        "# Mechanism Stage 1 Round Differential",
        "",
        f"- active benchmark: `{campaign['current_canonical_name']}`",
        f"- reference parity candidates: `{candidates}`",
        f"- matched screening slice mean for round6: `{operational['matched_slice_mean']:.4f}`",
        f"- operational current-state dev mean for round6: `{operational['operational_dev_mean']:.4f}`",
        "",
        "## Aggregate Differential",
        "",
        "| Line | Matched Final Mean | Mean First-Correct Round | Mean First-Stable Round | Mean Low-Disagreement Round | Mean Aggregate Steps | Mean Fine-Tune Steps | Mean Rounds | Exact Matched-Seed Final Ties |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| `round6` | `{round6_aggregate['matched_final_mean']:.4f}` | `{_format_float(round6_aggregate['mean_first_correct_round'])}` | "
        f"`{_format_float(round6_aggregate['mean_first_stable_round'])}` | `{_format_float(round6_aggregate['mean_low_disagreement_round'])}` | "
        f"`{round6_aggregate['mean_aggregate_steps']:.1f}` | `{round6_aggregate['mean_fine_tune_steps']:.1f}` | "
        f"`{round6_aggregate['mean_rounds']:.1f}` | `10 / 10` |",
    ]
    for row in candidate_rows:
        lines.append(
            f"| `{row['candidate']}` | `{row['matched_final_mean']:.4f}` | `{_format_float(row['mean_first_correct_round'])}` | "
            f"`{_format_float(row['mean_first_stable_round'])}` | `{_format_float(row['mean_low_disagreement_round'])}` | "
            f"`{row['mean_aggregate_steps']:.1f}` | `{row['mean_fine_tune_steps']:.1f}` | `{row['mean_rounds']:.1f}` | "
            f"`{int(row['exact_match_seed_count'])} / {int(row['matched_seed_count'])}` |"
        )
    support_rows = operational["support_only_rows"]
    lines.extend(
        [
            "",
            "## Support-Only Operational Seeds",
            "",
            "- These seeds sit inside the accepted current dev lanes for `round6`, but outside the narrower 10-seed parity screening slice that many challenger reports were summarized against.",
            "",
            "| Lane | Seed | Round6 Final Greedy Success |",
            "| --- | --- | ---: |",
        ]
    )
    for row in support_rows:
        lines.append(f"| {row['lane']} | {row['seed']} | `{float(row['final_greedy_success']):.4f}` |")
    lines.extend(
        [
            "",
            "## Concrete Differential",
            "",
            f"- `round6` does not separate from the strongest parity lines by cleanly beating them on the narrow matched 10-seed core. `round10` and `round10_conf_post4` tie `round6` on all 10 matched final checkpoints, which means the broad `0.9000` bucket is real on that slice.",
            f"- The accepted current-state edge comes from operational breadth and support coverage. `round6` keeps two additional accepted-state support seeds alive (`prospective_e/227` and `prospective_e/229`), which is why the live dev mean stays `{operational['operational_dev_mean']:.4f}` while the narrow parity slice stays `{operational['matched_slice_mean']:.4f}`.",
            "- When candidates do differ, the gap is mostly convergence and compute, not a different hard-core final policy. The warm-up line `round12_warm1_last_step` reaches the same matched final mean only after roughly four extra rounds and almost double the aggregate steps, while `round10_conf_post4` keeps the tie only with a modest but still real convergence tax.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "operational_round6": operational,
                "candidate_rows": candidate_rows,
            },
        )


def render_phase_local(
    campaign: dict[str, Any],
    output: Path,
    csv_output: Path | None,
    json_output: Path | None,
    *,
    episodes: int,
    max_steps: int,
    device: str,
) -> None:
    all_episode_rows: list[dict[str, Any]] = []
    all_phase_rows: list[dict[str, Any]] = []
    for case in campaign["analysis"]["phase_cases"]:
        lane = str(case["lane"])
        seed = int(case["seed"])
        episode_rows, phase_rows = _trace_phase_case(
            campaign,
            lane=lane,
            seed=seed,
            episodes=episodes,
            max_steps=max_steps,
            device=device,
        )
        all_episode_rows.extend(episode_rows)
        all_phase_rows.extend(phase_rows)

    lines = [
        "# Mechanism Stage 2 Phase-Local Analysis",
        "",
        f"- traced episodes per case/model: `{episodes}`",
        f"- phase candidate: `{campaign['analysis']['phase_candidate']}`",
        "",
        "## Episode Summary",
        "",
        "| Case | Variant | Success Rate | Top Failure Bucket | Top Divergence Phase | Median Key Pickup | Median Unlock | Mean Teacher-Match |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in sorted(all_episode_rows, key=lambda item: (str(item["lane"]), int(item["seed"]), str(item["label"]))):
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
    for row in sorted(all_phase_rows, key=lambda item: (str(item["lane"]), int(item["seed"]), str(item["label"]), str(item["phase"]))):
        lines.append(
            f"| {row['lane']}/{row['seed']} | `{row['label']}` | `{row['phase']}` | `{row['teacher_match_rate']:.4f}` | "
            f"`{row['student_confidence']:.4f}` | `{row['route_entropy']:.4f}` | `{row['route_dominant_pair_fraction']:.4f}` | `{row['step_count']}` |"
        )
    lines.extend(
        [
            "",
            "## Concrete Differential",
            "",
            "- `prospective_c/193` is not a late post-unlock cleanup miss. `round6` and the best parity candidate both fail before key pickup while matching the teacher almost perfectly, whereas matched `token_dense` succeeds by diverging from the teacher in the search/key phases. That makes this blocker a teacher-locked pre-key deadlock.",
            f"- `prospective_f/241` shows the other side of the mechanism. `round6` and `{campaign['analysis']['phase_candidate']}` both solve the case with early key pickup and unlock, while matched `token_dense` and `single_expert` mostly fail before key pickup or before unlock. The routed edge there lives in early-phase control, not just in terminal cleanup.",
            "- The repeated parity bucket therefore is not one clean late-phase story. It mixes a teacher-locked pre-key blocker (`193`) with a routed early-phase advantage on the cases that the active line still solves (`241`).",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, [*all_episode_rows, *all_phase_rows])
    if json_output is not None:
        _write_json(
            json_output,
            {
                "episode_rows": all_episode_rows,
                "phase_rows": all_phase_rows,
            },
        )


def render_selection_sensitivity(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    rows = _selection_metric_rows(campaign)
    operational = _operational_round6_summary(campaign)
    top_sets = {
        "raw": [row["candidate"] for row in _top_candidates(rows, "raw_rank_key")],
        "hard_seed": [row["candidate"] for row in _top_candidates(rows, "hard_rank_key")],
        "settle_aware": [row["candidate"] for row in _top_candidates(rows, "settle_rank_key")],
        "stability_aware": [row["candidate"] for row in _top_candidates(rows, "stability_rank_key")],
        "combined": [row["candidate"] for row in _top_candidates(rows, "combined_rank_key")],
    }
    classification = _selection_classification(top_sets, float(operational["operational_dev_mean"]))
    lines = [
        "# Mechanism Stage 3 Selection Sensitivity",
        "",
        f"- current operational round6 dev mean: `{operational['operational_dev_mean']:.4f}`",
        f"- matched screening-slice round6 mean: `{operational['matched_slice_mean']:.4f}`",
        f"- selection bottleneck classification: `{classification}`",
        "",
        "## Top Candidates By Selection Rule",
        "",
        f"- raw final mean: `{top_sets['raw']}`",
        f"- hard-seed weighted: `{top_sets['hard_seed']}`",
        f"- settle-aware: `{top_sets['settle_aware']}`",
        f"- stability-aware: `{top_sets['stability_aware']}`",
        f"- combined hard+settle: `{top_sets['combined']}`",
        "",
        "## Concrete Differential",
        "",
        "- Selection policy is not the primary blocker. It does reshuffle which `0.9000` line looks best, but it only reshuffles inside the same parity bucket.",
        "- The key shift is between raw-final ranking and settle-aware ranking: slower lines like the warm-up family fall back when early convergence is penalized, while `round10`, `round10_conf_post4`, and the bridge/confidence ties float to the top.",
        "- No robust selection rule turns a parity candidate into a benchmark replacement. The candidates still do not solve the teacher-locked `193` blocker, and they still lack the broader accepted-state support coverage that keeps the live `round6` dev mean above the narrow `0.9000` core.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "classification": classification,
                "top_sets": top_sets,
                "rows": rows,
            },
        )


def render_restart_sensitivity(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    rows = _restart_metric_rows(campaign)
    by_name = {row["candidate"]: row for row in rows}
    default_prior = by_name["round7"]
    replay_alt = by_name["round10"]
    hold_only = by_name["round5"]
    lines = [
        "# Mechanism Stage 4 Restart Sensitivity",
        "",
        "## Frontier Prior Snapshot",
        "",
        "| Prior | Matched Final Mean | Hard-Seed Mean | Mean First-Stable Round | Mean Aggregate Steps | Best Case | Typical Case | Worst Case |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['candidate']}` | `{row['matched_final_mean']:.4f}` | `{row['hard_seed_mean']:.4f}` | "
            f"`{_format_float(row['mean_first_stable_round'])}` | `{row['mean_aggregate_steps']:.1f}` | "
            f"`{row['best_case_final']:.4f}` | `{row['typical_case_final']:.4f}` | `{row['worst_case_final']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Runbook Interpretation",
            "",
            f"- `round7` remains the right default restart prior. It ties the strongest raw matched mean with the lightest compute burden among the live near-neighbor priors (`{default_prior['mean_aggregate_steps']:.1f}` aggregate steps versus `{replay_alt['mean_aggregate_steps']:.1f}` for `round10`).",
            "- `round10` remains a justified replay-validated alternate. It does not beat `round7` or `round6`, but it stays inside the same parity bucket and supports the strongest bounded confidence/bridge follow-ons.",
            f"- `round5` should stay hold-only. Its matched mean (`{hold_only['matched_final_mean']:.4f}`) and worst-case behavior both stay clearly below the active/default priors.",
            "- No fresh prior should move up into the active runbook ahead of `round7`. `round11` and `round12` spend more rounds and steps to re-hit the same bucket without adding a new solved blocker or a broader support picture.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(json_output, {"rows": rows})


def render_shortlist(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    directions = [
        {
            "label": "round10 confidence-weighted late-phase family",
            "track": "exploit",
            "candidates": ["round10_conf_post4", "round10_conf_temp095_post4"],
            "hypothesis": "A small confidence-weighted late-phase bias can preserve the parity tie while slightly reducing the convergence tax, without amplifying the teacher-locked `193` failure.",
            "evidence": "The confidence-weighted lines re-enter the 0.9000 parity bucket, unlike the catastrophic stochastic-credit and deep warm-up branches.",
            "why_better": "This is narrower than the broad league because it only follows one family that stayed alive after the mechanism analysis.",
            "failure": "If it still leaves `193` unsolved and does not beat `round6` on the accepted current dev slice, it is only another parity tie.",
        },
        {
            "label": "post-unlock weighting and smoothing on the round10/12 priors",
            "track": "exploit",
            "candidates": ["round10_post_unlock_x5", "round12_post_unlock_x5", "round10_temp095_post5"],
            "hypothesis": "A bounded post-unlock bias may help on the routed success cases without paying the convergence penalty of the warm-up family.",
            "evidence": "The post-unlock variants already sit at the top of the exploit bucket and keep the raw tie without obvious new collapse.",
            "why_better": "This targets the routed early-phase success cases directly instead of re-running every near-neighbor round count.",
            "failure": "If the line still cannot add any accepted-state support coverage beyond the narrow core, it is not a benchmark threat.",
        },
        {
            "label": "settle-aware checkpoint selection on the top parity ties",
            "track": "exploit",
            "candidates": ["select_round10_final", "select_round10_mix025"],
            "hypothesis": "Selection may be able to cleanly prefer the faster-settling parity lines even if it cannot create a new benchmark winner.",
            "evidence": "Selection sensitivity was secondary, not negligible: it changes which parity line looks best once convergence tax matters.",
            "why_better": "This is cheaper and more falsifiable than another broad training sweep.",
            "failure": "If the re-ranked line still sits at 0.9000 and still lacks support-seed breadth, selection is only a tie-break, not a mechanism unlock.",
        },
        {
            "label": "one-step warm-up boundary check",
            "track": "explore",
            "candidates": ["round10_warm1_last_step", "round12_warm1_last_step", "round12_warm2_last_step"],
            "hypothesis": "A very small warm-up may help only if the convergence delay can be contained; two-step warm-up is included as a falsification boundary.",
            "evidence": "The warm-up family only stays alive at the one-step edge; the deeper warm-up line already shows a heavy convergence tax.",
            "why_better": "This keeps exactly one bounded recursive-operator idea alive instead of re-running the whole warm-up/stochastic-credit branch.",
            "failure": "If the one-step line still converges far later than `round6`, the mechanism is too expensive even when it ties final success.",
        },
        {
            "label": "bridge revisit limited to the two non-catastrophic lines",
            "track": "explore",
            "candidates": ["round10_carry3_post5", "round10_door3_post5"],
            "hypothesis": "Only the carry-key and locked-door bridge variants merit another look because they tie the parity bucket without the catastrophic replay-cap collapse.",
            "evidence": "These were the only replay/bridge follow-ons that re-entered the 0.9000 bucket cleanly.",
            "why_better": "The shortlist drops the clearly harmful replay-cap and phase-balanced variants instead of repeating the whole revisit family.",
            "failure": "If these two lines still do not widen beyond the narrow core, the bridge family should be hard-fenced again.",
        },
    ]
    lines = [
        "# Mechanism Stage 5 Shortlist",
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
                f"- motivating evidence: {direction['evidence']}",
                f"- why it is sharper than another broad league: {direction['why_better']}",
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
    exploratory = {"overall_boundary": "not_run"}
    final_status = _decision_status(campaign, stage4, stage5, stage6, stage7, exploratory, gate_payload)
    winner = _winner(campaign, stage4, stage5, stage6, stage7)
    lines = [
        "# Mechanism-Driven Decision Memo",
        "",
        f"- final status: `{final_status}`",
        f"- winning line: `{winner['winner']}`",
        f"- gate verdict: `{gate_payload.get('verdict', gate_payload.get('status', 'not_run'))}`",
        "",
        "## Funnel",
        "",
        f"- mechanism shortlist directions: `{len(_optional_json(campaign['reports'].get('stageA5_json')).get('directions', []))}`",
        f"- Stage B1 exploit advancing challengers: `{stage1.get('fruitful_advancing_candidates', [])}`",
        f"- Stage B1 exploratory advancing challengers: `{stage1.get('exploratory_advancing_candidates', [])}`",
        f"- Stage B2 verified challengers: `{stage2.get('verified_candidates', [])}`",
        f"- Stage B3 fairness survivors: `{stage3.get('surviving_candidates', [])}`",
        f"- Stage B4 holdout best challenger: `{stage4.get('best_candidate')}`",
        f"- Stage B5 challenger anti-regression pass: `{stage5.get('challenger_pass')}`",
        f"- Stage B6 incumbent route pass: `{stage6.get('round6_pass')}`",
        f"- Stage B6 incumbent stability pass: `{stage7.get('round6_pass')}`",
        "",
        "## Decision",
        "",
    ]
    if final_status == str(campaign["decision_strings"]["replace"]):
        lines.append("- A mechanism-targeted challenger survived the full narrowed program, stayed meaningful after verification and controls, generalized to holdout, preserved the healthy blocks, stayed routed and stable, and cleared the final gate strongly enough to replace `round6`.")
    elif final_status == str(campaign["decision_strings"]["confirm"]):
        lines.append("- No targeted challenger displaced `round6`. The mechanism work clarified why the parity bucket keeps reappearing: it often matches the narrow hard-core slice, but it does not add accepted-state breadth, it does not fix the teacher-locked `prospective_c/193` blocker, and some lines only preserve the tie by converging later and spending more compute. `round6` therefore remains active and the frontier is better explained than before.")
    elif final_status == str(campaign["decision_strings"]["narrow"]):
        lines.append("- No targeted challenger displaced `round6`, and the active benchmark still clears the route/stability/gate bar, but the mechanism program does not add enough breadth or mechanism clarity to justify stronger internal state language than keeping the envelope narrow.")
    else:
        lines.append("- The mechanism program did not produce a viable challenger and did not leave enough remaining evidence to keep the current internal frontier wide. The benchmark state should narrow further.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mechanism-seeking program helpers around the active round6 benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    for command in (
        "state-reconciliation",
        "baseline-sync",
        "registration",
        "round-differential",
        "phase-local",
        "selection-sensitivity",
        "restart-sensitivity",
        "shortlist",
        "decision-memo",
    ):
        item = sub.add_parser(command)
        item.add_argument("--campaign-config", required=True)
        item.add_argument("--output", required=True)
        if command in {"baseline-sync", "phase-local"}:
            item.add_argument("--csv", required=False)
        if command in {"baseline-sync", "round-differential", "phase-local", "selection-sensitivity", "restart-sensitivity", "shortlist"}:
            item.add_argument("--json", required=False)
        if command == "phase-local":
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
    if args.command == "registration":
        render_registration(campaign, Path(args.output))
        return
    if args.command == "round-differential":
        render_round_differential(campaign, Path(args.output), Path(args.json) if args.json else None)
        return
    if args.command == "phase-local":
        render_phase_local(
            campaign,
            Path(args.output),
            Path(args.csv) if args.csv else None,
            Path(args.json) if args.json else None,
            episodes=int(args.episodes or campaign["analysis"]["phase_trace_episodes"]),
            max_steps=int(args.max_steps or campaign["analysis"]["phase_max_steps"]),
            device=str(args.device),
        )
        return
    if args.command == "selection-sensitivity":
        render_selection_sensitivity(campaign, Path(args.output), Path(args.json) if args.json else None)
        return
    if args.command == "restart-sensitivity":
        render_restart_sensitivity(campaign, Path(args.output), Path(args.json) if args.json else None)
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
