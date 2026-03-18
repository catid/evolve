from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psmn_rl.analysis.benchmark_pack import sha256_path
from psmn_rl.analysis.lss_hard_family_saturation import (
    _candidate_metric_row,
    _dedupe_metric_rows,
    _load_train_run_summary,
    _metric_block,
    _pack_metric_rows,
)
from psmn_rl.analysis.lss_post_pass_campaign import (
    DISPLAY_NAMES,
    LABELS,
    _artifact,
    _float,
    _int,
    _load_yaml,
    _read_csv_rows,
    _read_json,
    _stats,
    _summary_round,
    _write_csv,
    _write_json,
)
from psmn_rl.analysis.lss_robustness import _format_float
from psmn_rl.utils.io import get_git_commit, get_git_dirty


CURRENT_LABELS: tuple[str, ...] = ("kl_lss_sare", "kl_lss_token_dense", "kl_lss_single_expert")


@dataclass(slots=True)
class RunRecord:
    candidate: str
    lane: str
    seed: int
    label: str
    run_dir: Path
    summary: dict[str, Any]


def _block_specs(campaign: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [dict(block) for block in campaign["blocks"][key]]


def _block_lanes(campaign: dict[str, Any], key: str) -> tuple[str, ...]:
    return tuple(str(block["lane"]) for block in _block_specs(campaign, key))


def _lane_seed_pairs(campaign: dict[str, Any], key: str) -> list[tuple[str, int]]:
    return [(str(block["lane"]), int(seed)) for block in _block_specs(campaign, key) for seed in block.get("seeds", [])]


def _discover_runs(root: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    if not root.exists():
        return records
    for summary_path in sorted(root.glob("*/*/seed_*/kl_lss_*/summary.json")):
        run_dir = summary_path.parent
        seed_dir = run_dir.parent
        lane_dir = seed_dir.parent
        candidate_dir = lane_dir.parent
        if not seed_dir.name.startswith("seed_"):
            continue
        records.append(
            RunRecord(
                candidate=candidate_dir.name,
                lane=lane_dir.name,
                seed=int(seed_dir.name.split("_", 1)[1]),
                label=run_dir.name,
                run_dir=run_dir,
                summary=_read_json(summary_path),
            )
        )
    return records


def _candidate_row(record: RunRecord) -> dict[str, Any]:
    best_round_index = _int(record.summary.get("best_round_index"))
    best_round = _summary_round(record.summary, best_round_index)
    final_round = _summary_round(record.summary, len(record.summary.get("rounds", [])))
    return {
        "candidate": record.candidate,
        "lane": record.lane,
        "seed": record.seed,
        "label": record.label,
        "run_dir": str(record.run_dir),
        "config_path": str(record.run_dir / "student_resolved_config.yaml"),
        "checkpoint_path": str(record.run_dir / "latest.pt"),
        "final_greedy_success": _float(record.summary.get("final_greedy_success")),
        "best_round_index": best_round_index,
        "best_round_greedy_success": _float(record.summary.get("best_round_greedy_success")),
        "best_round_disagreement": _float(best_round.get("collection/disagreement_rate")),
        "best_round_post_unlock_frac": _float(best_round.get("collection/phase_frac_post_unlock")),
        "best_round_route_entropy": _float(best_round.get("collection/route_entropy")),
        "best_round_path_entropy": _float(best_round.get("collection/path_entropy")),
        "final_round_disagreement": _float(final_round.get("collection/disagreement_rate")),
        "final_round_post_unlock_frac": _float(final_round.get("collection/phase_frac_post_unlock")),
        "final_round_route_entropy": _float(final_round.get("collection/route_entropy")),
        "final_round_path_entropy": _float(final_round.get("collection/path_entropy")),
        "final_round_active_compute": _float(final_round.get("collection/active_compute_proxy")),
        "final_round_aggregate_steps": _float(final_round.get("aggregate_steps")),
        "final_round_fine_tune_steps": _float(final_round.get("fine_tune/steps")),
    }


def _current_round6_rows(campaign: dict[str, Any], keys: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    lane_filter = None if keys is None else set(keys)
    rows: list[dict[str, Any]] = []
    sare_roots = campaign["current_round6_sare_roots"]
    control_roots = campaign["current_round6_control_roots"]
    for lane, sare_root_str in sare_roots.items():
        if lane_filter is not None and str(lane) not in lane_filter:
            continue
        sare_root = Path(sare_root_str)
        control_root = Path(control_roots[str(lane)])
        for seed_dir in sorted(sare_root.glob("seed_*")):
            if not seed_dir.name.startswith("seed_"):
                continue
            seed = int(seed_dir.name.split("_", 1)[1])
            for label, root in (
                ("kl_lss_sare", sare_root),
                ("kl_lss_token_dense", control_root),
                ("kl_lss_single_expert", control_root),
            ):
                run_dir = root / f"seed_{seed}" / label
                summary_path = run_dir / "summary.json"
                if not summary_path.exists():
                    continue
                summary = _read_json(summary_path)
                rows.append(
                    {
                        "candidate": str(campaign["current_canonical_name"]),
                        "lane": str(lane),
                        "seed": seed,
                        "label": label,
                        "run_dir": str(run_dir),
                        "config_path": str(run_dir / "student_resolved_config.yaml"),
                        "checkpoint_path": str(run_dir / "latest.pt"),
                        "summary": summary,
                        "final_greedy_success": _float(summary.get("final_greedy_success")),
                        "best_round_index": _int(summary.get("best_round_index")),
                        "best_round_greedy_success": _float(summary.get("best_round_greedy_success")),
                    }
                )
    return rows


def _run_dir(root: Path, lane: str, seed: int, label: str = "kl_lss_sare") -> Path:
    return root / lane / f"seed_{seed}" / label


def _round6_run_dir(campaign: dict[str, Any], lane: str, seed: int, label: str = "kl_lss_sare") -> Path:
    if label == "kl_lss_sare":
        root = Path(campaign["current_round6_sare_roots"][lane])
    else:
        root = Path(campaign["current_round6_control_roots"][lane])
    return root / f"seed_{seed}" / label


def _candidate_stage_root(campaign: dict[str, Any], stage_key: str, candidate: str) -> Path:
    return Path(campaign["stage_roots"][stage_key]) / candidate


def _candidate_run_dir(campaign: dict[str, Any], stage_key: str, candidate: str, lane: str, seed: int, label: str = "kl_lss_sare") -> Path:
    return _candidate_stage_root(campaign, stage_key, candidate) / lane / f"seed_{seed}" / label


def _split_stats(rows: list[dict[str, Any]], candidate: str, label: str, lanes: tuple[str, ...]) -> dict[str, float]:
    values = [
        _float(row["final_greedy_success"])
        for row in rows
        if str(row["candidate"]) == candidate and str(row["label"]) == label and str(row["lane"]) in lanes
    ]
    return _stats(values)


def _block_summary(rows: list[dict[str, Any]], candidate: str, label: str, lanes: tuple[str, ...]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for lane in lanes:
        values = [
            _float(row["final_greedy_success"])
            for row in rows
            if str(row["candidate"]) == candidate and str(row["label"]) == label and str(row["lane"]) == lane
        ]
        result.append(
            {
                "candidate": candidate,
                "lane": lane,
                "label": label,
                **_stats(values),
            }
        )
    return result


def _control_family(rows: list[dict[str, Any]], candidate: str, lanes: tuple[str, ...]) -> dict[str, float]:
    sare = _split_stats(rows, candidate, "kl_lss_sare", lanes)
    token = _split_stats(rows, candidate, "kl_lss_token_dense", lanes)
    single = _split_stats(rows, candidate, "kl_lss_single_expert", lanes)
    return {
        "sare_mean": sare["mean"],
        "token_mean": token["mean"],
        "single_mean": single["mean"],
        "sare_failures": sare["complete_seed_failures"],
        "sare_minus_token": sare["mean"] - token["mean"],
        "sare_minus_single": sare["mean"] - single["mean"],
    }


def _stability_class(values: list[float]) -> str:
    if not values:
        return "no_data"
    sorted_values = sorted(values, reverse=True)
    best = sorted_values[0]
    second = sorted_values[1] if len(sorted_values) > 1 else sorted_values[0]
    final_value = values[-1]
    if (best - second) <= 0.125 and (best - final_value) <= 0.125:
        return "stable_plateau"
    if (best - second) > 0.25 or (best - final_value) > 0.25:
        return "narrow_spike"
    return "noisy_brittle"


def _summary_curve(run_dir: Path) -> list[float]:
    summary = _read_json(run_dir / "summary.json")
    return [_float(round_row.get("after_greedy_success")) for round_row in summary.get("rounds", [])]


def _stability_summary(run_dir: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / "summary.json")
    curve = _summary_curve(run_dir)
    return {
        "run_dir": str(run_dir),
        "curve": curve,
        "best_round_index": _int(summary.get("best_round_index")),
        "best_round_greedy_success": _float(summary.get("best_round_greedy_success")),
        "final_greedy_success": _float(summary.get("final_greedy_success")),
        "stability_class": _stability_class(curve),
    }


def _route_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(row for row in rows if str(row["probe"]) == "baseline")
    baseline_success = _float(baseline.get("eval_success_rate"))
    fixed = next(row for row in rows if str(row["probe"]) == "router_override")
    randomization = next(row for row in rows if str(row["probe"]) == "route_randomization")
    ablations = [row for row in rows if str(row["probe"]) == "expert_ablation"]
    worst = min(ablations, key=lambda item: _float(item.get("eval_success_rate"))) if ablations else baseline
    return {
        "lane": str(baseline["lane"]),
        "seed": int(baseline["seed"]),
        "baseline_success": baseline_success,
        "fixed_router_drop": baseline_success - _float(fixed.get("eval_success_rate")),
        "route_randomization_drop": baseline_success - _float(randomization.get("eval_success_rate")),
        "worst_ablation_drop": baseline_success - _float(worst.get("eval_success_rate")),
        "worst_ablation_expert": str(worst.get("detail")),
        "baseline_route_entropy": _float(baseline.get("route_entropy")),
        "baseline_path_entropy": _float(baseline.get("path_entropy")),
        "baseline_active_compute": _float(baseline.get("active_compute_proxy")),
    }


def _route_pass(campaign: dict[str, Any], summaries: list[dict[str, Any]]) -> bool:
    if not summaries:
        return False
    fixed_ok = all(_float(row["fixed_router_drop"]) >= _float(campaign["selection"]["route_drop_min"]) for row in summaries)
    ablation_ok = all(_float(row["worst_ablation_drop"]) >= _float(campaign["selection"]["route_drop_min"]) for row in summaries)
    random_ok = sum(1 for row in summaries if _float(row["route_randomization_drop"]) >= _float(campaign["selection"]["route_randomization_drop_min"])) >= 2
    return fixed_ok and ablation_ok and random_ok


def _historical_refs(campaign: dict[str, Any]) -> list[str]:
    return [str(path) for path in campaign.get("historical_reports", [])]


def _render_registration(campaign: dict[str, Any], output: Path) -> None:
    directions = sorted({str(meta["direction"]) for meta in campaign["candidates"].values()})
    lines = [
        "# Successor Migration Registration",
        "",
        f"- archived frozen benchmark pack: `{campaign['legacy_frozen_pack']}`",
        f"- current qualified successor pack: `{campaign['current_canonical_pack']}`",
        f"- current qualified successor: `{campaign['current_canonical_name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Splits",
        "",
        f"- challenger development families: `{_block_specs(campaign, 'dev')}`",
        f"- challenger holdout families: `{_block_specs(campaign, 'holdout')}`",
        f"- healthy anti-regression families: `{_block_specs(campaign, 'healthy')}`",
        "",
        "## Challenger League",
        "",
        f"- mechanism directions: `{directions}`",
        f"- total challenger variants: `{len(campaign['candidates'])}`",
        "",
        "## Migration Rule",
        "",
        "- If no challenger survives the full league cleanly, the program still continues through route, stability, and pack/gate checks for round6 before deciding active canonization vs no migration.",
        "- If a challenger survives the full league and still beats or equals round6 after controls, holdout, anti-regression, route, and stability, the challenger is packaged for migration instead.",
        "",
        "## Historical Context",
        "",
    ]
    for path in _historical_refs(campaign):
        lines.append(f"- `{path}`")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_baseline_sync(campaign: dict[str, Any], output: Path, csv_output: Path | None, json_output: Path | None) -> None:
    frozen_pack = _read_json(Path(campaign["frozen_pack"]))
    current_pack = _read_json(Path(campaign["current_canonical_pack"]))
    current_rows = _current_round6_rows(campaign)
    dev_lanes = _block_lanes(campaign, "dev")
    holdout_lanes = _block_lanes(campaign, "holdout")
    healthy_lanes = _block_lanes(campaign, "healthy")
    dev = _control_family(current_rows, str(campaign["current_canonical_name"]), dev_lanes)
    holdout = _control_family(current_rows, str(campaign["current_canonical_name"]), holdout_lanes)
    healthy = _control_family(current_rows, str(campaign["current_canonical_name"]), healthy_lanes)
    lines = [
        "# Successor Migration Baseline Sync",
        "",
        f"- archived frozen benchmark pack: `{campaign['legacy_frozen_pack']}`",
        f"- current qualified successor pack: `{campaign['current_canonical_pack']}`",
        f"- current qualified successor: `{campaign['current_canonical_name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Frozen Gate Thresholds",
        "",
        f"- retry-block KL learner-state `SARE` mean to beat: `{_float(frozen_pack['claim']['thresholds']['retry_block']['kl_lss_sare_mean']):.4f}`" if "claim" in frozen_pack and "thresholds" in frozen_pack["claim"] else f"- frozen pack reference: `{campaign['frozen_pack']}`",
        "",
        "## Current Qualified Successor Snapshot",
        "",
        f"- retry-block KL learner-state `SARE` mean: `{_float(current_pack['metrics']['retry_block']['kl_lss_sare']['mean']):.4f}`",
        f"- retry-block KL learner-state `single_expert` mean: `{_float(current_pack['metrics']['retry_block']['kl_lss_single_expert']['mean']):.4f}`",
        f"- frozen-comparable combined KL learner-state `SARE` mean: `{_float(current_pack['metrics']['combined']['kl_lss_sare']['mean']):.4f}`",
        f"- dev-family round6 SARE/token/single: `{dev['sare_mean']:.4f}` / `{dev['token_mean']:.4f}` / `{dev['single_mean']:.4f}`",
        f"- holdout-family round6 SARE/token/single: `{holdout['sare_mean']:.4f}` / `{holdout['token_mean']:.4f}` / `{holdout['single_mean']:.4f}`",
        f"- healthy-family round6 SARE/token/single: `{healthy['sare_mean']:.4f}` / `{healthy['token_mean']:.4f}` / `{healthy['single_mean']:.4f}`",
        "",
        "## Prior Validation Memos",
        "",
    ]
    for path in _historical_refs(campaign):
        lines.append(f"- `{path}`")
    lines.extend(["", "## Current Per-Seed Snapshot", "", "| Block | Seed | Variant | Greedy Success |", "| --- | --- | --- | ---: |"])
    for row in sorted(
        [row for row in current_rows if str(row["label"]) in CURRENT_LABELS],
        key=lambda item: (str(item["lane"]), int(item["seed"]), str(item["label"])),
    ):
        lines.append(
            f"| {row['lane']} | {row['seed']} | {DISPLAY_NAMES[str(row['label'])]} | {_format_float(_float(row['final_greedy_success']))} |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, current_rows)
    if json_output is not None:
        _write_json(
            json_output,
            {
                "current_pack": current_pack,
                "dev": dev,
                "holdout": holdout,
                "healthy": healthy,
                "current_rows": current_rows,
                "historical_reports": _historical_refs(campaign),
            },
        )


def _stage1_pass(campaign: dict[str, Any], summary: dict[str, Any], incumbent_failures: float) -> bool:
    return (
        _float(summary["delta_vs_round6"]) >= _float(campaign["selection"]["min_dev_gain"])
        and _float(summary["candidate_failures"]) <= _float(incumbent_failures)
        and _float(summary["max_block_delta"]) >= _float(campaign["selection"]["min_dev_gain"])
    )


def _render_stage1(campaign: dict[str, Any], output: Path, csv_output: Path | None, json_output: Path | None) -> None:
    dev_lanes = _block_lanes(campaign, "dev")
    round6_rows = [row for row in _current_round6_rows(campaign, dev_lanes) if str(row["label"]) in CURRENT_LABELS]
    challenger_rows = [_candidate_row(record) for record in _discover_runs(Path(campaign["stage_roots"]["stage1_screening"])) if str(record.label) == "kl_lss_sare"]
    round6_family = _control_family(round6_rows, str(campaign["current_canonical_name"]), dev_lanes)
    screening_rows: list[dict[str, Any]] = []
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    round6_lookup = {(str(row["lane"]), int(row["seed"]), str(row["label"])): row for row in round6_rows}
    for row in sorted(challenger_rows, key=lambda item: (str(item["candidate"]), str(item["lane"]), int(item["seed"]))):
        token = round6_lookup[(str(row["lane"]), int(row["seed"]), "kl_lss_token_dense")]
        single = round6_lookup[(str(row["lane"]), int(row["seed"]), "kl_lss_single_expert")]
        incumbent = round6_lookup[(str(row["lane"]), int(row["seed"]), "kl_lss_sare")]
        record = {
            **row,
            "delta_vs_round6": _float(row["final_greedy_success"]) - _float(incumbent["final_greedy_success"]),
            "delta_vs_token_dense": _float(row["final_greedy_success"]) - _float(token["final_greedy_success"]),
            "delta_vs_single_expert": _float(row["final_greedy_success"]) - _float(single["final_greedy_success"]),
        }
        screening_rows.append(record)
        by_candidate.setdefault(str(row["candidate"]), []).append(record)
    candidate_summaries: list[dict[str, Any]] = []
    for candidate, rows in sorted(by_candidate.items()):
        stats = _stats([_float(row["final_greedy_success"]) for row in rows])
        block_deltas = []
        for lane in dev_lanes:
            lane_rows = [row for row in rows if str(row["lane"]) == lane]
            lane_mean = _stats([_float(row["final_greedy_success"]) for row in lane_rows])["mean"]
            round6_mean = _stats(
                [
                    _float(row["final_greedy_success"])
                    for row in round6_rows
                    if str(row["lane"]) == lane and str(row["label"]) == "kl_lss_sare"
                ]
            )["mean"]
            block_deltas.append(lane_mean - round6_mean)
        summary = {
            "candidate": candidate,
            "candidate_mean": stats["mean"],
            "round6_mean": round6_family["sare_mean"],
            "token_mean": round6_family["token_mean"],
            "single_mean": round6_family["single_mean"],
            "candidate_failures": stats["complete_seed_failures"],
            "delta_vs_round6": stats["mean"] - round6_family["sare_mean"],
            "candidate_minus_token": stats["mean"] - round6_family["token_mean"],
            "candidate_minus_single": stats["mean"] - round6_family["single_mean"],
            "max_block_delta": max(block_deltas) if block_deltas else 0.0,
            "mean_aggregate_steps": _stats([_float(row["final_round_aggregate_steps"]) for row in rows])["mean"],
            "mean_fine_tune_steps": _stats([_float(row["final_round_fine_tune_steps"]) for row in rows])["mean"],
        }
        summary["stage1_pass"] = _stage1_pass(campaign, summary, round6_family["sare_failures"])
        candidate_summaries.append(summary)
    advancing_candidates = [
        row["candidate"]
        for row in sorted(
            [row for row in candidate_summaries if row["stage1_pass"]],
            key=lambda item: (item["candidate_mean"], item["delta_vs_round6"], item["candidate_minus_token"]),
            reverse=True,
        )[: _int(campaign["selection"]["stage1_top_k"])]
    ]
    lines = [
        "# Successor Migration Stage 1 Challenger Screening",
        "",
        f"- incumbent round6 dev SARE/token/single: `{round6_family['sare_mean']:.4f}` / `{round6_family['token_mean']:.4f}` / `{round6_family['single_mean']:.4f}`",
        f"- advancing challengers: `{advancing_candidates}`",
        "",
        "| Candidate | Block | Seed | Greedy Success | Δ vs round6 | Δ vs token_dense | Δ vs single_expert | Best Round | Mean Aggregate Steps |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in screening_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['candidate']}`",
                    str(row["lane"]),
                    str(row["seed"]),
                    _format_float(row["final_greedy_success"]),
                    _format_float(row["delta_vs_round6"]),
                    _format_float(row["delta_vs_token_dense"]),
                    _format_float(row["delta_vs_single_expert"]),
                    str(_int(row["best_round_index"])),
                    _format_float(row["final_round_aggregate_steps"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Candidate Summary", "", "| Candidate | Dev Mean | Δ vs round6 | Candidate-token | Candidate-single | Failures | Mean Aggregate Steps | Stage 1 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in sorted(candidate_summaries, key=lambda item: item["candidate_mean"], reverse=True):
        lines.append(
            f"| `{row['candidate']}` | `{row['candidate_mean']:.4f}` | `{row['delta_vs_round6']:.4f}` | `{row['candidate_minus_token']:.4f}` | `{row['candidate_minus_single']:.4f}` | `{int(row['candidate_failures'])}` | `{row['mean_aggregate_steps']:.1f}` | `{'pass' if row['stage1_pass'] else 'stop'}` |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, screening_rows)
    if json_output is not None:
        _write_json(
            json_output,
            {
                "rows": screening_rows,
                "round6_summary": round6_family,
                "candidate_summaries": candidate_summaries,
                "advancing_candidates": advancing_candidates,
            },
        )


def _stage2_pass(campaign: dict[str, Any], round6: dict[str, float], challenger: dict[str, float]) -> bool:
    eps = _float(campaign["selection"]["control_eps"])
    margin_eps = _float(campaign["selection"]["margin_eps"])
    control_ok = challenger["sare_mean"] + eps >= challenger["token_mean"] and challenger["sare_mean"] + eps >= challenger["single_mean"]
    failures_ok = challenger["sare_failures"] <= round6["sare_failures"]
    meaningful = (
        challenger["sare_mean"] >= round6["sare_mean"] + margin_eps
        or challenger["sare_minus_token"] >= round6["sare_minus_token"] + margin_eps
        or challenger["sare_minus_single"] >= round6["sare_minus_single"] + margin_eps
    )
    return control_ok and failures_ok and meaningful


def _render_stage2(campaign: dict[str, Any], output: Path, csv_output: Path | None, json_output: Path | None) -> None:
    stage1 = _read_json(Path(campaign["reports"]["stage1_json"]))
    dev_lanes = _block_lanes(campaign, "dev")
    round6_rows = [row for row in _current_round6_rows(campaign, dev_lanes) if str(row["label"]) in CURRENT_LABELS]
    fairness_rows = [_candidate_row(record) for record in _discover_runs(Path(campaign["stage_roots"]["stage2_fairness"]))]
    candidate_names = [str(name) for name in stage1.get("advancing_candidates", [])]
    sare_rows = [row for row in stage1.get("rows", []) if str(row["candidate"]) in candidate_names]
    detail_rows = sare_rows + [row for row in fairness_rows if str(row["candidate"]) in candidate_names]
    round6_summary = _control_family(round6_rows, str(campaign["current_canonical_name"]), dev_lanes)
    challenger_summaries: list[dict[str, Any]] = []
    for candidate in candidate_names:
        candidate_rows = [row for row in detail_rows if str(row["candidate"]) == candidate]
        summary = _control_family(candidate_rows, candidate, dev_lanes)
        payload = {
            "candidate": candidate,
            **summary,
            "delta_vs_round6": summary["sare_mean"] - round6_summary["sare_mean"],
        }
        payload["stage2_pass"] = _stage2_pass(campaign, round6_summary, payload)
        challenger_summaries.append(payload)
    surviving_candidates = [
        row["candidate"]
        for row in sorted(
            [row for row in challenger_summaries if row["stage2_pass"]],
            key=lambda item: (item["sare_mean"], item["sare_minus_token"], item["delta_vs_round6"]),
            reverse=True,
        )[: _int(campaign["selection"]["stage2_top_k"])]
    ]
    lines = [
        "# Successor Migration Stage 2 Fairness",
        "",
        f"- surviving challengers: `{surviving_candidates}`",
        "",
        "| Candidate | Dev SARE | Dev token_dense | Dev single_expert | SARE-token | SARE-single | Δ vs round6 | Failures | Stage 2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        f"| `{campaign['current_canonical_name']}` | `{round6_summary['sare_mean']:.4f}` | `{round6_summary['token_mean']:.4f}` | `{round6_summary['single_mean']:.4f}` | `{round6_summary['sare_minus_token']:.4f}` | `{round6_summary['sare_minus_single']:.4f}` | `0.0000` | `{int(round6_summary['sare_failures'])}` | `incumbent` |",
    ]
    for row in sorted(challenger_summaries, key=lambda item: item["sare_mean"], reverse=True):
        lines.append(
            f"| `{row['candidate']}` | `{row['sare_mean']:.4f}` | `{row['token_mean']:.4f}` | `{row['single_mean']:.4f}` | `{row['sare_minus_token']:.4f}` | `{row['sare_minus_single']:.4f}` | `{row['delta_vs_round6']:.4f}` | `{int(row['sare_failures'])}` | `{'pass' if row['stage2_pass'] else 'stop'}` |"
        )
    lines.extend(["", "| Candidate | Block | Seed | Variant | Greedy Success |", "| --- | --- | --- | --- | ---: |"])
    for row in sorted(round6_rows + detail_rows, key=lambda item: (str(item["candidate"]), str(item["lane"]), int(item["seed"]), str(item["label"]))):
        if str(row["candidate"]) not in (str(campaign["current_canonical_name"]),) + tuple(candidate_names):
            continue
        if str(row["label"]) not in CURRENT_LABELS:
            continue
        lines.append(f"| `{row['candidate']}` | {row['lane']} | {row['seed']} | {DISPLAY_NAMES[str(row['label'])]} | {_format_float(_float(row['final_greedy_success']))} |")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, round6_rows + detail_rows)
    if json_output is not None:
        _write_json(
            json_output,
            {
                "round6_summary": round6_summary,
                "detail_rows": detail_rows,
                "surviving_candidates": surviving_candidates,
                "challenger_summaries": challenger_summaries,
            },
        )


def _stage3_pass(campaign: dict[str, Any], round6: dict[str, float], challenger: dict[str, float]) -> bool:
    eps = _float(campaign["selection"]["control_eps"])
    min_holdout_gain = _float(campaign["selection"]["min_holdout_gain"])
    return (
        challenger["sare_mean"] + eps >= challenger["token_mean"]
        and challenger["sare_mean"] + eps >= challenger["single_mean"]
        and challenger["sare_mean"] + eps >= round6["sare_mean"]
        and challenger["sare_failures"] <= round6["sare_failures"]
        and challenger["delta_vs_round6"] >= -min_holdout_gain
    )


def _render_stage3(campaign: dict[str, Any], output: Path, csv_output: Path | None, json_output: Path | None) -> None:
    stage2 = _read_json(Path(campaign["reports"]["stage2_json"]))
    holdout_lanes = _block_lanes(campaign, "holdout")
    round6_rows = [row for row in _current_round6_rows(campaign, holdout_lanes) if str(row["label"]) in CURRENT_LABELS]
    challenger_names = [str(name) for name in stage2.get("surviving_candidates", [])]
    detail_rows = [_candidate_row(record) for record in _discover_runs(Path(campaign["stage_roots"]["stage3_holdout"])) if str(record.candidate) in challenger_names]
    round6_summary = _control_family(round6_rows, str(campaign["current_canonical_name"]), holdout_lanes)
    challenger_summaries: list[dict[str, Any]] = []
    for candidate in challenger_names:
        candidate_rows = [row for row in detail_rows if str(row["candidate"]) == candidate]
        summary = _control_family(candidate_rows, candidate, holdout_lanes)
        payload = {"candidate": candidate, **summary, "delta_vs_round6": summary["sare_mean"] - round6_summary["sare_mean"]}
        payload["stage3_pass"] = _stage3_pass(campaign, round6_summary, payload)
        challenger_summaries.append(payload)
    surviving = [
        row["candidate"]
        for row in sorted(
            [row for row in challenger_summaries if row["stage3_pass"]],
            key=lambda item: (item["sare_mean"], item["sare_minus_token"], item["delta_vs_round6"]),
            reverse=True,
        )
    ]
    best_candidate = surviving[0] if surviving else None
    lines = [
        "# Successor Migration Stage 3 Holdout",
        "",
        f"- best surviving challenger: `{best_candidate}`",
        "",
        "| Candidate | Holdout SARE | Holdout token_dense | Holdout single_expert | SARE-token | SARE-single | Δ vs round6 | Failures | Stage 3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        f"| `{campaign['current_canonical_name']}` | `{round6_summary['sare_mean']:.4f}` | `{round6_summary['token_mean']:.4f}` | `{round6_summary['single_mean']:.4f}` | `{round6_summary['sare_minus_token']:.4f}` | `{round6_summary['sare_minus_single']:.4f}` | `0.0000` | `{int(round6_summary['sare_failures'])}` | `incumbent` |",
    ]
    for row in sorted(challenger_summaries, key=lambda item: item["sare_mean"], reverse=True):
        lines.append(
            f"| `{row['candidate']}` | `{row['sare_mean']:.4f}` | `{row['token_mean']:.4f}` | `{row['single_mean']:.4f}` | `{row['sare_minus_token']:.4f}` | `{row['sare_minus_single']:.4f}` | `{row['delta_vs_round6']:.4f}` | `{int(row['sare_failures'])}` | `{'pass' if row['stage3_pass'] else 'stop'}` |"
        )
    lines.extend(["", "| Candidate | Block | Seed | Variant | Greedy Success |", "| --- | --- | --- | --- | ---: |"])
    for row in sorted(round6_rows + detail_rows, key=lambda item: (str(item["candidate"]), str(item["lane"]), int(item["seed"]), str(item["label"]))):
        if str(row["candidate"]) not in (str(campaign["current_canonical_name"]),) + tuple(challenger_names):
            continue
        if str(row["label"]) not in CURRENT_LABELS:
            continue
        lines.append(f"| `{row['candidate']}` | {row['lane']} | {row['seed']} | {DISPLAY_NAMES[str(row['label'])]} | {_format_float(_float(row['final_greedy_success']))} |")
    if not best_candidate:
        lines.extend(["", "## Interpretation", "", "- No challenger survived holdout. The program now shifts fully to migration readiness for round6."])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, round6_rows + detail_rows)
    if json_output is not None:
        _write_json(
            json_output,
            {
                "round6_summary": round6_summary,
                "detail_rows": detail_rows,
                "challenger_summaries": challenger_summaries,
                "best_candidate": best_candidate,
                "surviving_candidates": surviving,
            },
        )


def _stage4_pass(campaign: dict[str, Any], round6: dict[str, float], challenger: dict[str, float]) -> bool:
    tol = _float(campaign["selection"]["anti_regression_tolerance"])
    return (
        challenger["sare_mean"] >= round6["sare_mean"] - tol
        and challenger["sare_failures"] <= round6["sare_failures"]
    )


def _render_stage4(campaign: dict[str, Any], output: Path, csv_output: Path | None, json_output: Path | None) -> None:
    stage3 = _read_json(Path(campaign["reports"]["stage3_json"]))
    healthy_lanes = _block_lanes(campaign, "healthy")
    round6_rows = [row for row in _current_round6_rows(campaign, healthy_lanes) if str(row["label"]) in CURRENT_LABELS]
    best_candidate = stage3.get("best_candidate")
    detail_rows = [
        _candidate_row(record)
        for record in _discover_runs(Path(campaign["stage_roots"]["stage4_antiregression"]))
        if best_candidate and str(record.candidate) == str(best_candidate)
    ]
    round6_summary = _control_family(round6_rows, str(campaign["current_canonical_name"]), healthy_lanes)
    challenger_summary = None
    challenger_pass = False
    if best_candidate:
        summary = _control_family(detail_rows, str(best_candidate), healthy_lanes)
        challenger_summary = {"candidate": str(best_candidate), **summary, "delta_vs_round6": summary["sare_mean"] - round6_summary["sare_mean"]}
        challenger_pass = _stage4_pass(campaign, round6_summary, challenger_summary)
    lines = [
        "# Successor Migration Stage 4 Anti-Regression",
        "",
        f"- best challenger entering anti-regression: `{best_candidate}`",
        "",
        "| Candidate | Healthy SARE | Healthy token_dense | Healthy single_expert | SARE-token | SARE-single | Δ vs round6 | Failures | Stage 4 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        f"| `{campaign['current_canonical_name']}` | `{round6_summary['sare_mean']:.4f}` | `{round6_summary['token_mean']:.4f}` | `{round6_summary['single_mean']:.4f}` | `{round6_summary['sare_minus_token']:.4f}` | `{round6_summary['sare_minus_single']:.4f}` | `0.0000` | `{int(round6_summary['sare_failures'])}` | `incumbent` |",
    ]
    if challenger_summary is not None:
        lines.append(
            f"| `{challenger_summary['candidate']}` | `{challenger_summary['sare_mean']:.4f}` | `{challenger_summary['token_mean']:.4f}` | `{challenger_summary['single_mean']:.4f}` | `{challenger_summary['sare_minus_token']:.4f}` | `{challenger_summary['sare_minus_single']:.4f}` | `{challenger_summary['delta_vs_round6']:.4f}` | `{int(challenger_summary['sare_failures'])}` | `{'pass' if challenger_pass else 'stop'}` |"
        )
    lines.extend(["", "| Candidate | Block | Seed | Variant | Greedy Success |", "| --- | --- | --- | --- | ---: |"])
    for row in sorted(round6_rows + detail_rows, key=lambda item: (str(item["candidate"]), str(item["lane"]), int(item["seed"]), str(item["label"]))):
        if str(row["candidate"]) not in (str(campaign["current_canonical_name"]), str(best_candidate)):
            continue
        if str(row["label"]) not in CURRENT_LABELS:
            continue
        lines.append(f"| `{row['candidate']}` | {row['lane']} | {row['seed']} | {DISPLAY_NAMES[str(row['label'])]} | {_format_float(_float(row['final_greedy_success']))} |")
    if best_candidate is None:
        lines.extend(["", "## Interpretation", "", "- No challenger reached anti-regression. This stage records the incumbent healthy-block picture for the migration decision."])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, round6_rows + detail_rows)
    if json_output is not None:
        _write_json(
            json_output,
            {
                "round6_summary": round6_summary,
                "detail_rows": detail_rows,
                "best_candidate": best_candidate,
                "challenger_summary": challenger_summary,
                "challenger_pass": challenger_pass,
            },
        )


def _route_csv_path(raw_dir: Path, line_name: str, case_name: str) -> Path:
    return raw_dir / f"{line_name}_{case_name}.csv"


def _route_md_path(raw_dir: Path, line_name: str, case_name: str) -> Path:
    return raw_dir / f"{line_name}_{case_name}.md"


def _route_case_names() -> tuple[str, ...]:
    return ("dev", "holdout", "healthy")


def _load_route_set(campaign: dict[str, Any], raw_dir: Path, line_name: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for case_name in _route_case_names():
        csv_path = _route_csv_path(raw_dir, line_name, case_name)
        if csv_path.exists():
            summary = _route_summary(_read_csv_rows(csv_path))
            summary["case"] = case_name
            result.append(summary)
    return result


def _render_stage5(campaign: dict[str, Any], output: Path, csv_output: Path | None, json_output: Path | None) -> None:
    stage3 = _read_json(Path(campaign["reports"]["stage3_json"]))
    stage4 = _read_json(Path(campaign["reports"]["stage4_json"]))
    raw_dir = Path(campaign["reports"]["route_raw_dir"])
    best_candidate = stage3.get("best_candidate") if stage4.get("challenger_pass") else None
    round6_routes = _load_route_set(campaign, raw_dir, str(campaign["current_canonical_name"]))
    challenger_routes = _load_route_set(campaign, raw_dir, str(best_candidate)) if best_candidate else []
    round6_pass = _route_pass(campaign, round6_routes)
    challenger_pass = _route_pass(campaign, challenger_routes) if challenger_routes else False
    lines = [
        "# Successor Migration Stage 5 Route Validation",
        "",
        f"- incumbent round6 route status: `{'pass' if round6_pass else 'fail'}`",
        f"- challenger route status: `{'pass' if challenger_pass else 'not_applicable' if not best_candidate else 'fail'}`",
        "",
        "| Line | Case | Baseline | Fixed-Router Drop | Randomization Drop | Worst-Ablation Drop | Route Entropy | Active Compute |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, rows in ((str(campaign["current_canonical_name"]), round6_routes), (str(best_candidate), challenger_routes)):
        for row in rows:
            lines.append(
                f"| `{label}` | {row['case']} | `{row['baseline_success']:.4f}` | `{row['fixed_router_drop']:.4f}` | `{row['route_randomization_drop']:.4f}` | `{row['worst_ablation_drop']:.4f}` | `{row['baseline_route_entropy']:.4f}` | `{row['baseline_active_compute']:.4f}` |"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_rows = [{"line": str(campaign["current_canonical_name"]), **row} for row in round6_routes] + [{"line": str(best_candidate), **row} for row in challenger_routes]
    if csv_output is not None:
        _write_csv(csv_output, csv_rows)
    if json_output is not None:
        _write_json(
            json_output,
            {
                "round6_routes": round6_routes,
                "challenger_routes": challenger_routes,
                "round6_pass": round6_pass,
                "challenger_pass": challenger_pass,
                "best_candidate": best_candidate,
            },
        )


def _render_stage6(campaign: dict[str, Any], output: Path, csv_output: Path | None, json_output: Path | None) -> None:
    stage3 = _read_json(Path(campaign["reports"]["stage3_json"]))
    stage4 = _read_json(Path(campaign["reports"]["stage4_json"]))
    best_candidate = stage3.get("best_candidate") if stage4.get("challenger_pass") else None
    case_map = campaign["stability_cases"]
    summaries: list[dict[str, Any]] = []
    for case_name in _route_case_names():
        lane = str(case_map[case_name]["lane"])
        seed = _int(case_map[case_name]["seed"])
        round6_summary = _stability_summary(_round6_run_dir(campaign, lane, seed))
        summaries.append({"line": str(campaign["current_canonical_name"]), "case": case_name, "lane": lane, "seed": seed, **round6_summary})
        if best_candidate:
            if case_name == "dev":
                run_dir = _candidate_run_dir(campaign, "stage1_screening", str(best_candidate), lane, seed)
            elif case_name == "holdout":
                run_dir = _candidate_run_dir(campaign, "stage3_holdout", str(best_candidate), lane, seed)
            else:
                run_dir = _candidate_run_dir(campaign, "stage4_antiregression", str(best_candidate), lane, seed)
            summaries.append({"line": str(best_candidate), "case": case_name, "lane": lane, "seed": seed, **_stability_summary(run_dir)})
    round6_pass = all(row["stability_class"] != "narrow_spike" for row in summaries if row["line"] == str(campaign["current_canonical_name"]))
    challenger_pass = bool(best_candidate) and all(row["stability_class"] != "narrow_spike" for row in summaries if row["line"] == str(best_candidate))
    lines = [
        "# Successor Migration Stage 6 Stability",
        "",
        f"- incumbent round6 stability status: `{'pass' if round6_pass else 'fail'}`",
        f"- challenger stability status: `{'pass' if challenger_pass else 'not_applicable' if not best_candidate else 'fail'}`",
        "",
        "| Line | Case | Best Round | Best Greedy | Final Greedy | Stability |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summaries:
        lines.append(
            f"| `{row['line']}` | {row['case']} | `{row['best_round_index']}` | `{row['best_round_greedy_success']:.4f}` | `{row['final_greedy_success']:.4f}` | `{row['stability_class']}` |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, summaries)
    if json_output is not None:
        _write_json(
            json_output,
            {
                "summaries": summaries,
                "best_candidate": best_candidate,
                "round6_pass": round6_pass,
                "challenger_pass": challenger_pass,
            },
        )


def _winner(campaign: dict[str, Any], stage3: dict[str, Any], stage4: dict[str, Any], stage5: dict[str, Any], stage6: dict[str, Any]) -> dict[str, Any]:
    best_candidate = stage3.get("best_candidate")
    challenger_viable = (
        best_candidate is not None
        and bool(stage4.get("challenger_pass"))
        and bool(stage5.get("challenger_pass"))
        and bool(stage6.get("challenger_pass"))
    )
    return {
        "winner": str(best_candidate) if challenger_viable else str(campaign["current_canonical_name"]),
        "challenger_viable": challenger_viable,
        "best_candidate": best_candidate,
    }


def _fresh_final_candidate_rows(campaign: dict[str, Any], candidate: str) -> list[dict[str, Any]]:
    root = Path(campaign["stage_roots"]["stage7_pack"]) / candidate
    return [_candidate_row(record) for record in _discover_runs(root) if str(record.lane) == "fresh_final"]


def _combined_pack_rows(campaign: dict[str, Any], candidate: str, stage4: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_combined_rows = [row for row in _read_csv_rows(Path("outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.csv")) if str(row.get("mode")) == "greedy"]
    baseline_rows = [
        {
            "lane": str(row["lane"]),
            "seed": int(row["seed"]),
            "label": str(row["label"]),
            "eval_success_rate": _float(row["eval_success_rate"]),
            "run_dir": str(row["run_dir"]),
        }
        for row in baseline_combined_rows
        if str(row["label"]) in {"recovered_token_dense", "baseline_sare"}
    ]
    healthy_rows = [_candidate_metric_row(row) for row in stage4.get("detail_rows", []) if str(row.get("candidate")) == candidate and str(row["label"]) in CURRENT_LABELS]
    retry_rows = [_candidate_metric_row(row) for row in _fresh_final_candidate_rows(campaign, candidate) if str(row["label"]) in CURRENT_LABELS]
    combined_lane_seeds = _lane_seed_pairs(campaign, "frozen_combined")
    retry_lane_seeds = _lane_seed_pairs({"blocks": {"retry": [{"lane": "fresh_final", "seeds": [47, 53, 59]}]}}, "retry")
    combined_rows = _dedupe_metric_rows(_pack_metric_rows(healthy_rows + retry_rows + baseline_rows, combined_lane_seeds))
    retry_pack_rows = _dedupe_metric_rows(_pack_metric_rows(retry_rows + baseline_rows, retry_lane_seeds))
    return combined_rows, retry_pack_rows


def _render_candidate_reports(campaign: dict[str, Any], candidate: str, stage3: dict[str, Any], stage4: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    reports = campaign["reports"]
    combined_rows, retry_rows = _combined_pack_rows(campaign, candidate, stage4)
    combined_metrics = _metric_block(combined_rows)
    retry_metrics = _metric_block(retry_rows)
    dev_summary = next((row for row in stage3.get("challenger_summaries", []) if str(row["candidate"]) == candidate), None)
    holdout_summary = dev_summary if candidate == str(campaign["current_canonical_name"]) else next((row for row in stage3.get("challenger_summaries", []) if str(row["candidate"]) == candidate), None)
    summary_lines = [
        "# Successor Migration Candidate Summary",
        "",
        f"- candidate: `{candidate}`",
        f"- retry-block KL learner-state `SARE` mean: `{retry_metrics['kl_lss_sare']['mean']:.4f}`",
        f"- frozen-comparable combined KL learner-state `SARE` mean: `{combined_metrics['kl_lss_sare']['mean']:.4f}`",
    ]
    Path(reports["candidate_summary_markdown"]).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    combined_lines = [
        "# Successor Migration Candidate Combined Report",
        "",
        "| Lane | Seed | Variant | Greedy Success |",
        "| --- | --- | --- | ---: |",
    ]
    for row in sorted(combined_rows, key=lambda item: (str(item["lane"]), int(item["seed"]), str(item["label"]))):
        combined_lines.append(
            f"| {row['lane']} | {row['seed']} | {DISPLAY_NAMES[str(row['label'])]} | {_format_float(_float(row['eval_success_rate']))} |"
        )
    Path(reports["candidate_combined_markdown"]).write_text("\n".join(combined_lines) + "\n", encoding="utf-8")
    _write_csv(Path(reports["candidate_combined_csv"]), combined_rows)
    retry_lines = [
        "# Successor Migration Candidate Retry Report",
        "",
        "| Seed | recovered token_dense | KL learner-state token_dense | KL learner-state single_expert | baseline PPO SARE | KL learner-state SARE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lookup = {(str(row["lane"]), int(row["seed"]), str(row["label"])): row for row in retry_rows}
    for seed in (47, 53, 59):
        retry_lines.append(
            "| "
            + " | ".join(
                [
                    str(seed),
                    _format_float(lookup[("fresh_final", seed, "recovered_token_dense")]["eval_success_rate"]),
                    _format_float(lookup[("fresh_final", seed, "kl_lss_token_dense")]["eval_success_rate"]),
                    _format_float(lookup[("fresh_final", seed, "kl_lss_single_expert")]["eval_success_rate"]),
                    _format_float(lookup[("fresh_final", seed, "baseline_sare")]["eval_success_rate"]),
                    _format_float(lookup[("fresh_final", seed, "kl_lss_sare")]["eval_success_rate"]),
                ]
            )
            + " |"
        )
    Path(reports["candidate_retry_markdown"]).write_text("\n".join(retry_lines) + "\n", encoding="utf-8")
    _write_csv(Path(reports["candidate_retry_csv"]), retry_rows)
    metrics_payload = {
        "schema_version": 1,
        "candidate_name": candidate,
        "task": "DoorKey",
        "evaluation": campaign["evaluation"],
        "requested_claims": ["bounded_teacher_guided_doorkey_sare"],
        "controls_present": list(LABELS),
        "metrics": {"combined": combined_metrics, "retry_block": retry_metrics},
        "actual_sets": {
            "combined_lane_seeds": [[lane, seed] for lane, seed in _lane_seed_pairs(campaign, "frozen_combined")],
            "retry_block_lane_seeds": [["fresh_final", 47], ["fresh_final", 53], ["fresh_final", 59]],
        },
        "provenance": {
            "git_commit": get_git_commit(),
            "git_dirty": get_git_dirty(),
            "notes": f"migration challenger `{candidate}`",
        },
    }
    _write_json(Path(reports["candidate_metrics_json"]), metrics_payload)
    return metrics_payload, combined_rows, retry_rows


def _round6_migration_pack(campaign: dict[str, Any], stage3: dict[str, Any], stage4: dict[str, Any], stage5: dict[str, Any], stage6: dict[str, Any]) -> dict[str, Any]:
    current_pack = copy.deepcopy(_read_json(Path(campaign["current_canonical_pack"])))
    frozen_pack = _read_json(Path(campaign["frozen_pack"]))
    current_pack["frozen_pack_reference"] = {
        "path": str(campaign["frozen_pack"]),
        "sha256": sha256_path(Path(campaign["frozen_pack"])),
        "claim_id": frozen_pack["claim"]["id"],
    }
    current_pack["migration"] = {
        "decision": "round6 canonized as active DoorKey benchmark",
        "active_canonical_candidate": str(campaign["current_canonical_name"]),
        "archived_legacy_frozen_pack": {
            "path": str(campaign["legacy_frozen_pack"]),
            "sha256": sha256_path(Path(campaign["legacy_frozen_pack"])),
        },
        "current_source_pack": {
            "path": str(campaign["current_canonical_pack"]),
            "sha256": sha256_path(Path(campaign["current_canonical_pack"])),
        },
        "future_comparison_policy": {
            "active_canonical_pack": "successor_migration_candidate_pack.json",
            "archived_legacy_pack": str(campaign["legacy_frozen_pack"]),
        },
        "route_validation": stage5,
        "stability_validation": stage6,
        "holdout_summary": stage3.get("round6_summary"),
        "healthy_summary": stage4.get("round6_summary"),
    }
    current_pack.setdefault("provenance", {})
    current_pack["provenance"]["git_commit"] = get_git_commit()
    current_pack["provenance"]["git_dirty"] = get_git_dirty()
    current_pack["provenance"]["migration_notes"] = "round6 promoted to active DoorKey benchmark draft while frozen pack remains archived"
    return current_pack


def _challenger_migration_pack(campaign: dict[str, Any], candidate: str, stage3: dict[str, Any], stage4: dict[str, Any], stage5: dict[str, Any], stage6: dict[str, Any]) -> dict[str, Any]:
    metrics_payload, _, _ = _render_candidate_reports(campaign, candidate, stage3, stage4)
    pack = {
        "schema_version": 1,
        "pack_type": "candidate_result_pack",
        "candidate_name": candidate,
        "frozen_pack_reference": {
            "path": str(campaign["frozen_pack"]),
            "sha256": sha256_path(Path(campaign["frozen_pack"])),
            "claim_id": _read_json(Path(campaign["frozen_pack"]))["claim"]["id"],
        },
        "task": "DoorKey",
        "evaluation": campaign["evaluation"],
        "requested_claims": ["bounded_teacher_guided_doorkey_sare"],
        "controls_present": list(LABELS),
        "metrics": metrics_payload["metrics"],
        "actual_sets": metrics_payload["actual_sets"],
        "artifacts": [
            _artifact(Path(campaign["reports"]["candidate_summary_markdown"]), "candidate_summary_markdown"),
            _artifact(Path(campaign["reports"]["candidate_metrics_json"]), "candidate_metrics_json"),
            _artifact(Path(campaign["reports"]["candidate_combined_markdown"]), "combined_report_markdown"),
            _artifact(Path(campaign["reports"]["candidate_combined_csv"]), "combined_report_csv"),
            _artifact(Path(campaign["reports"]["candidate_retry_markdown"]), "retry_block_report_markdown"),
            _artifact(Path(campaign["reports"]["candidate_retry_csv"]), "retry_block_report_csv"),
        ],
        "provenance": metrics_payload["provenance"],
        "migration": {
            "decision": "challenger canonized as active DoorKey benchmark",
            "replaces_round6": True,
            "previous_active_pack": {
                "path": str(campaign["current_canonical_pack"]),
                "sha256": sha256_path(Path(campaign["current_canonical_pack"])),
            },
            "archived_legacy_frozen_pack": {
                "path": str(campaign["legacy_frozen_pack"]),
                "sha256": sha256_path(Path(campaign["legacy_frozen_pack"])),
            },
            "route_validation": stage5,
            "stability_validation": stage6,
            "holdout_summary": next(row for row in stage3.get("challenger_summaries", []) if str(row["candidate"]) == candidate),
            "healthy_summary": stage4.get("challenger_summary"),
        },
    }
    return pack


def _render_migration_pack(campaign: dict[str, Any], output: Path) -> None:
    stage3 = _read_json(Path(campaign["reports"]["stage3_json"]))
    stage4 = _read_json(Path(campaign["reports"]["stage4_json"]))
    stage5 = _read_json(Path(campaign["reports"]["stage5_json"]))
    stage6 = _read_json(Path(campaign["reports"]["stage6_json"]))
    winner = _winner(campaign, stage3, stage4, stage5, stage6)
    if winner["winner"] == str(campaign["current_canonical_name"]):
        pack = _round6_migration_pack(campaign, stage3, stage4, stage5, stage6)
    else:
        pack = _challenger_migration_pack(campaign, str(winner["winner"]), stage3, stage4, stage5, stage6)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pack, indent=2, sort_keys=True), encoding="utf-8")


def _render_decision_memo(campaign: dict[str, Any], output: Path) -> None:
    stage1 = _read_json(Path(campaign["reports"]["stage1_json"]))
    stage2 = _read_json(Path(campaign["reports"]["stage2_json"]))
    stage3 = _read_json(Path(campaign["reports"]["stage3_json"]))
    stage4 = _read_json(Path(campaign["reports"]["stage4_json"]))
    stage5 = _read_json(Path(campaign["reports"]["stage5_json"]))
    stage6 = _read_json(Path(campaign["reports"]["stage6_json"]))
    gate_payload = _read_json(Path(campaign["reports"]["gate_report_json"])) if Path(campaign["reports"]["gate_report_json"]).exists() else {"status": "not_run"}
    winner = _winner(campaign, stage3, stage4, stage5, stage6)
    gate_verdict = str(gate_payload.get("verdict", gate_payload.get("status", "not_run")))
    round6_ready = bool(stage5.get("round6_pass")) and bool(stage6.get("round6_pass")) and gate_verdict == str(campaign["selection"]["pack_gate_required_verdict"])
    if winner["winner"] != str(campaign["current_canonical_name"]) and gate_verdict == str(campaign["selection"]["pack_gate_required_verdict"]):
        final_status = "challenger canonized as active DoorKey benchmark"
    elif winner["winner"] == str(campaign["current_canonical_name"]) and round6_ready:
        final_status = "round6 canonized as active DoorKey benchmark"
    elif winner["winner"] != str(campaign["current_canonical_name"]):
        final_status = "challenger fails and round6 remains thaw-qualified but not canonical"
    else:
        final_status = "round6 remains thaw-qualified but not canonical"
    lines = [
        "# Successor Migration Decision Memo",
        "",
        f"- final status: `{final_status}`",
        f"- winning line: `{winner['winner']}`",
        f"- gate verdict: `{gate_verdict}`",
        "",
        "## Challenger League Summary",
        "",
        f"- Stage 1 advancing challengers: `{stage1.get('advancing_candidates', [])}`",
        f"- Stage 2 surviving challengers: `{stage2.get('surviving_candidates', [])}`",
        f"- Stage 3 best challenger: `{stage3.get('best_candidate')}`",
        f"- Stage 4 challenger anti-regression pass: `{stage4.get('challenger_pass')}`",
        f"- Stage 5 incumbent route pass: `{stage5.get('round6_pass')}`",
        f"- Stage 5 challenger route pass: `{stage5.get('challenger_pass')}`",
        f"- Stage 6 incumbent stability pass: `{stage6.get('round6_pass')}`",
        f"- Stage 6 challenger stability pass: `{stage6.get('challenger_pass')}`",
        "",
        "## Decision",
        "",
    ]
    if final_status == "challenger canonized as active DoorKey benchmark":
        lines.append("- A within-family challenger beat the incumbent cleanly enough after controls, holdout, anti-regression, route, stability, and gate checks to take the active DoorKey benchmark role.")
    elif final_status == "round6 canonized as active DoorKey benchmark":
        lines.append("- No challenger displaced round6 cleanly enough. Round6 stayed strong across the challenger league and now becomes the active DoorKey benchmark while the frozen pack remains archived as the legacy baseline.")
    elif final_status == "challenger fails and round6 remains thaw-qualified but not canonical":
        lines.append("- A challenger reached the later stages but did not clear the final migration bar, and round6 also did not earn the stronger migration outcome in this run.")
    else:
        lines.append("- No challenger displaced round6, but the migration program stops short of active canonization, so round6 remains thaw-qualified only.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-canonization migration and challenger league for DoorKey")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("registration", "baseline-sync", "stage1-report", "stage2-report", "stage3-report", "stage4-report", "stage5-report", "stage6-report", "migration-pack", "decision-memo"):
        stage = sub.add_parser(name)
        stage.add_argument("--campaign-config", required=True)
        if name != "migration-pack":
            stage.add_argument("--output", required=True)
        if name in {"baseline-sync", "stage1-report", "stage2-report", "stage3-report", "stage4-report", "stage5-report", "stage6-report"}:
            stage.add_argument("--csv", required=False)
            stage.add_argument("--json", required=False)
        if name == "migration-pack":
            stage.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    campaign = _load_yaml(Path(args.campaign_config))

    if args.command == "registration":
        _render_registration(campaign, Path(args.output))
        return
    if args.command == "baseline-sync":
        _render_baseline_sync(
            campaign,
            Path(args.output),
            Path(args.csv) if args.csv else None,
            Path(args.json) if args.json else None,
        )
        return
    if args.command == "stage1-report":
        _render_stage1(campaign, Path(args.output), Path(args.csv) if args.csv else None, Path(args.json) if args.json else None)
        return
    if args.command == "stage2-report":
        _render_stage2(campaign, Path(args.output), Path(args.csv) if args.csv else None, Path(args.json) if args.json else None)
        return
    if args.command == "stage3-report":
        _render_stage3(campaign, Path(args.output), Path(args.csv) if args.csv else None, Path(args.json) if args.json else None)
        return
    if args.command == "stage4-report":
        _render_stage4(campaign, Path(args.output), Path(args.csv) if args.csv else None, Path(args.json) if args.json else None)
        return
    if args.command == "stage5-report":
        _render_stage5(campaign, Path(args.output), Path(args.csv) if args.csv else None, Path(args.json) if args.json else None)
        return
    if args.command == "stage6-report":
        _render_stage6(campaign, Path(args.output), Path(args.csv) if args.csv else None, Path(args.json) if args.json else None)
        return
    if args.command == "migration-pack":
        _render_migration_pack(campaign, Path(args.output))
        return
    if args.command == "decision-memo":
        _render_decision_memo(campaign, Path(args.output))
        return
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
