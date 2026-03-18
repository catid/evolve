from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_hard_family_saturation import _stability_class
from psmn_rl.analysis.lss_post_pass_campaign import (
    DISPLAY_NAMES,
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
CHALLENGER_ORDER: tuple[str, ...] = ("round7", "round8")


@dataclass(slots=True)
class RunRecord:
    candidate: str
    lane: str
    seed: int
    label: str
    run_dir: Path
    summary: dict[str, Any]


def _block_lanes(campaign: dict[str, Any], key: str) -> tuple[str, ...]:
    return tuple(str(block["lane"]) for block in campaign["blocks"][key])


def _stage1_candidate_run_dir(campaign: dict[str, Any], candidate: str, lane: str, seed: int) -> Path:
    return Path(campaign["stage_roots"]["stage1_screening"]) / candidate / lane / f"seed_{seed}" / "kl_lss_sare"


def _current_round6_run_dir(campaign: dict[str, Any], lane: str, seed: int, label: str = "kl_lss_sare") -> Path:
    return Path(campaign["current_round6_roots"][lane]) / f"seed_{seed}" / label


def _round_greedy_curve(summary: dict[str, Any]) -> list[float]:
    return [_float(round_row.get("after_greedy_success")) for round_row in summary.get("rounds", [])]


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
        "final_greedy_success": _float(record.summary.get("final_greedy_success")),
        "best_round_index": best_round_index,
        "best_round_greedy_success": _float(record.summary.get("best_round_greedy_success")),
        "best_round_disagreement": _float(best_round.get("collection/disagreement_rate")),
        "best_round_post_unlock_frac": _float(best_round.get("collection/phase_frac_post_unlock")),
        "final_round_disagreement": _float(final_round.get("collection/disagreement_rate")),
        "final_round_post_unlock_frac": _float(final_round.get("collection/phase_frac_post_unlock")),
        "final_round_route_entropy": _float(final_round.get("collection/route_entropy")),
        "final_round_path_entropy": _float(final_round.get("collection/path_entropy")),
    }


def _load_current_round6_rows(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane, root_str in campaign["current_round6_roots"].items():
        root = Path(root_str)
        for summary_path in sorted(root.glob("seed_*/kl_lss_*/summary.json")):
            run_dir = summary_path.parent
            seed_dir = run_dir.parent
            if not seed_dir.name.startswith("seed_"):
                continue
            rows.append(
                {
                    "candidate": str(campaign["current_canonical_name"]),
                    "lane": str(lane),
                    "seed": int(seed_dir.name.split("_", 1)[1]),
                    "label": run_dir.name,
                    "run_dir": str(run_dir),
                    "summary": _read_json(summary_path),
                }
            )
    return [
        {
            **row,
            **{
                "final_greedy_success": _float(row["summary"].get("final_greedy_success")),
                "best_round_index": _int(row["summary"].get("best_round_index")),
                "best_round_greedy_success": _float(row["summary"].get("best_round_greedy_success")),
            },
        }
        for row in rows
    ]


def _split_stats(rows: list[dict[str, Any]], candidate: str, label: str, lanes: tuple[str, ...]) -> dict[str, float]:
    values = [
        _float(row["final_greedy_success"])
        for row in rows
        if str(row["candidate"]) == candidate and str(row["label"]) == label and str(row["lane"]) in lanes
    ]
    return _stats(values)


def _stage1_candidate_summary(campaign: dict[str, Any], rows: list[dict[str, Any]], candidate: str) -> dict[str, float]:
    hard_dev_lanes = _block_lanes(campaign, "hard_dev")
    hard_holdout_lanes = _block_lanes(campaign, "hard_holdout")
    healthy_lanes = _block_lanes(campaign, "healthy")
    hard_dev = _split_stats(rows, candidate, "kl_lss_sare", hard_dev_lanes)
    hard_holdout = _split_stats(rows, candidate, "kl_lss_sare", hard_holdout_lanes)
    healthy = _split_stats(rows, candidate, "kl_lss_sare", healthy_lanes)
    hard_all = _stats(
        [
            _float(row["final_greedy_success"])
            for row in rows
            if str(row["candidate"]) == candidate
            and str(row["label"]) == "kl_lss_sare"
            and str(row["lane"]) in hard_dev_lanes + hard_holdout_lanes
        ]
    )
    return {
        "hard_dev_mean": hard_dev["mean"],
        "hard_holdout_mean": hard_holdout["mean"],
        "hard_combined_mean": hard_all["mean"],
        "healthy_mean": healthy["mean"],
        "hard_combined_failures": hard_all["complete_seed_failures"],
        "healthy_failures": healthy["complete_seed_failures"],
    }


def _select_best_challenger(summaries: dict[str, dict[str, float]]) -> str:
    return max(
        CHALLENGER_ORDER,
        key=lambda candidate: (
            summaries[candidate]["hard_combined_mean"],
            summaries[candidate]["hard_holdout_mean"],
            summaries[candidate]["healthy_mean"],
            -summaries[candidate]["hard_combined_failures"],
            -summaries[candidate]["healthy_failures"],
            -CHALLENGER_ORDER.index(candidate),
        ),
    )


def _challenger_fairness_summary(
    campaign: dict[str, Any],
    combined_rows: list[dict[str, Any]],
    stage1_summaries: dict[str, dict[str, float]],
    candidate: str,
) -> dict[str, float]:
    hard_dev_lanes = _block_lanes(campaign, "hard_dev")
    hard_holdout_lanes = _block_lanes(campaign, "hard_holdout")
    hard_lanes = hard_dev_lanes + hard_holdout_lanes
    hard_sare = _split_stats(combined_rows, candidate, "kl_lss_sare", hard_lanes)
    hard_token = _split_stats(combined_rows, candidate, "kl_lss_token_dense", hard_lanes)
    hard_single = _split_stats(combined_rows, candidate, "kl_lss_single_expert", hard_lanes)
    holdout_sare = _split_stats(combined_rows, candidate, "kl_lss_sare", hard_holdout_lanes)
    return {
        "hard_combined_sare": hard_sare["mean"],
        "hard_combined_token": hard_token["mean"],
        "hard_combined_single": hard_single["mean"],
        "hard_sare_minus_token": hard_sare["mean"] - hard_token["mean"],
        "hard_sare_minus_single": hard_sare["mean"] - hard_single["mean"],
        "hard_failures": hard_sare["complete_seed_failures"],
        "hard_holdout_sare": holdout_sare["mean"],
        "healthy_mean": stage1_summaries[candidate]["healthy_mean"],
    }


def _promote_challenger(
    campaign: dict[str, Any],
    incumbent: dict[str, float],
    challenger: dict[str, float],
) -> bool:
    eps = _float(campaign["selection"]["equality_eps"])
    if challenger["healthy_mean"] + eps < incumbent["healthy_mean"]:
        return False
    if challenger["hard_combined_sare"] > incumbent["hard_combined_sare"] + eps:
        return (
            challenger["hard_combined_sare"] + eps >= challenger["hard_combined_token"]
            and challenger["hard_combined_sare"] + eps >= challenger["hard_combined_single"]
        )
    if abs(challenger["hard_combined_sare"] - incumbent["hard_combined_sare"]) <= eps:
        if challenger["hard_combined_sare"] + eps < challenger["hard_combined_token"]:
            return False
        if challenger["hard_combined_sare"] + eps < challenger["hard_combined_single"]:
            return False
        if challenger["hard_sare_minus_token"] > incumbent["hard_sare_minus_token"] + eps:
            return True
        if challenger["hard_sare_minus_single"] > incumbent["hard_sare_minus_single"] + eps:
            return True
    return False


def _route_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(row for row in rows if str(row["probe"]) == "baseline")
    baseline_success = _float(baseline.get("eval_success_rate"))
    fixed = next(row for row in rows if str(row["probe"]) == "router_override")
    randomization = next(row for row in rows if str(row["probe"]) == "route_randomization")
    ablations = [row for row in rows if str(row["probe"]) == "expert_ablation"]
    worst_ablation = min(ablations, key=lambda row: _float(row.get("eval_success_rate")))
    return {
        "lane": str(baseline["lane"]),
        "seed": int(baseline["seed"]),
        "baseline_success": baseline_success,
        "fixed_router_drop": baseline_success - _float(fixed.get("eval_success_rate")),
        "route_randomization_drop": baseline_success - _float(randomization.get("eval_success_rate")),
        "worst_ablation_drop": baseline_success - _float(worst_ablation.get("eval_success_rate")),
        "worst_ablation_expert": str(worst_ablation.get("detail")),
        "baseline_route_entropy": _float(baseline.get("route_entropy")),
        "baseline_active_compute": _float(baseline.get("active_compute_proxy")),
    }


def _route_pass(campaign: dict[str, Any], summary: dict[str, Any]) -> bool:
    return (
        summary["fixed_router_drop"] >= _float(campaign["selection"]["route_drop_min"])
        and summary["worst_ablation_drop"] >= _float(campaign["selection"]["route_drop_min"])
        and summary["route_randomization_drop"] >= _float(campaign["selection"]["route_randomization_drop_min"])
    )


def _stability_summary(run_dir: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / "summary.json")
    curve = _round_greedy_curve(summary)
    return {
        "run_dir": str(run_dir),
        "curve": curve,
        "best_round_index": _int(summary.get("best_round_index")),
        "best_round_greedy_success": _float(summary.get("best_round_greedy_success")),
        "final_greedy_success": _float(summary.get("final_greedy_success")),
        "stability_class": _stability_class(curve),
    }


def _render_registration(campaign: dict[str, Any], output: Path) -> None:
    lines = [
        "# Successor Carryover Challenge Registration",
        "",
        f"- frozen benchmark pack: `{campaign['frozen_pack']}`",
        f"- current canonical successor pack: `{campaign['current_canonical_pack']}`",
        f"- current canonical successor: `{campaign['current_canonical_name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Blocks",
        "",
        f"- hard-family development blocks: `{campaign['blocks']['hard_dev']}`",
        f"- hard-family holdout blocks: `{campaign['blocks']['hard_holdout']}`",
        f"- healthy carryover blocks: `{campaign['blocks']['healthy']}`",
        "",
        "## Challengers",
        "",
    ]
    for candidate, meta in campaign["candidates"].items():
        lines.append(f"- `{candidate}`: {meta['description']}")
    lines.extend(
        [
            "",
            "## Plan",
            "",
            "- Stage 1: run challenger SARE lines across hard-family carryover blocks and one healthy carryover block.",
            "- Stage 2: add matched token_dense and single_expert controls on the hard-family carryover split.",
            "- Stage 3: route- and stability-check the best challenger before deciding whether it displaces round6.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_stage1(campaign: dict[str, Any], output: Path, csv_output: Path | None = None, json_output: Path | None = None) -> None:
    challenger_rows = [_candidate_row(record) for record in _discover_runs(Path(campaign["stage_roots"]["stage1_screening"]))]
    current_rows = [
        row
        for row in _load_current_round6_rows(campaign)
        if str(row["label"]) == "kl_lss_sare"
    ]
    rows = current_rows + challenger_rows
    candidate_names = (str(campaign["current_canonical_name"]),) + CHALLENGER_ORDER
    summaries = {candidate: _stage1_candidate_summary(campaign, rows, candidate) for candidate in candidate_names}
    selected_challenger = _select_best_challenger(summaries)
    lines = [
        "# Successor Carryover Challenge Stage 1 Screening",
        "",
        f"- selected challenger: `{selected_challenger}`",
        "",
        "| Candidate | Hard Dev SARE | Hard Holdout SARE | Hard Combined SARE | Healthy SARE | Hard Failures | Healthy Failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in candidate_names:
        summary = summaries[candidate]
        lines.append(
            f"| `{candidate}` | `{summary['hard_dev_mean']:.4f}` | `{summary['hard_holdout_mean']:.4f}` | `{summary['hard_combined_mean']:.4f}` | `{summary['healthy_mean']:.4f}` | `{int(summary['hard_combined_failures'])}` | `{int(summary['healthy_failures'])}` |"
        )
    lines.extend(["", "| Candidate | Block | Seed | Greedy Success |", "| --- | --- | --- | ---: |"])
    for row in sorted(rows, key=lambda item: (str(item["candidate"]), str(item["lane"]), int(item["seed"]))):
        lines.append(f"| `{row['candidate']}` | {row['lane']} | {row['seed']} | {_format_float(_float(row['final_greedy_success']))} |")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, rows)
    if json_output is not None:
        _write_json(json_output, {"rows": rows, "candidate_summaries": summaries, "selected_challenger": selected_challenger})


def _render_stage2(campaign: dict[str, Any], output: Path, csv_output: Path | None = None, json_output: Path | None = None) -> None:
    stage1 = _read_json(Path(campaign["reports"]["stage1_json"]))
    incumbent_rows = [
        row
        for row in _load_current_round6_rows(campaign)
        if str(row["lane"]) in _block_lanes(campaign, "hard_dev") + _block_lanes(campaign, "hard_holdout")
    ]
    fairness_rows = [_candidate_row(record) for record in _discover_runs(Path(campaign["stage_roots"]["stage2_fairness"]))]
    challenger_sare_rows = [
        row
        for row in stage1["rows"]
        if str(row["candidate"]) in CHALLENGER_ORDER
        and str(row["lane"]) in _block_lanes(campaign, "hard_dev") + _block_lanes(campaign, "hard_holdout")
    ]
    combined_rows = incumbent_rows + challenger_sare_rows + fairness_rows
    incumbent_summary = _challenger_fairness_summary(
        campaign,
        combined_rows,
        stage1["candidate_summaries"],
        str(campaign["current_canonical_name"]),
    )
    selected_challenger = _select_best_challenger(stage1["candidate_summaries"])
    challenger_summaries = {
        selected_challenger: _challenger_fairness_summary(
            campaign,
            combined_rows,
            stage1["candidate_summaries"],
            selected_challenger,
        )
    }
    promote = _promote_challenger(campaign, incumbent_summary, challenger_summaries[selected_challenger])
    lines = [
        "# Successor Carryover Challenge Stage 2 Fairness",
        "",
        f"- selected challenger: `{selected_challenger}`",
        f"- challenger promotion status: `{'promote' if promote else 'keep_round6'}`",
        "",
        "| Candidate | Hard Combined SARE | Hard token_dense | Hard single_expert | SARE-token | SARE-single | Hard Holdout SARE | Healthy SARE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| `{campaign['current_canonical_name']}` | `{incumbent_summary['hard_combined_sare']:.4f}` | "
            f"`{incumbent_summary['hard_combined_token']:.4f}` | `{incumbent_summary['hard_combined_single']:.4f}` | "
            f"`{incumbent_summary['hard_sare_minus_token']:.4f}` | `{incumbent_summary['hard_sare_minus_single']:.4f}` | "
            f"`{incumbent_summary['hard_holdout_sare']:.4f}` | `{incumbent_summary['healthy_mean']:.4f}` |"
        ),
    ]
    for candidate in (selected_challenger,):
        summary = challenger_summaries[candidate]
        lines.append(
            f"| `{candidate}` | `{summary['hard_combined_sare']:.4f}` | `{summary['hard_combined_token']:.4f}` | `{summary['hard_combined_single']:.4f}` | `{summary['hard_sare_minus_token']:.4f}` | `{summary['hard_sare_minus_single']:.4f}` | `{summary['hard_holdout_sare']:.4f}` | `{summary['healthy_mean']:.4f}` |"
        )
    lines.extend(["", "| Candidate | Block | Seed | Variant | Greedy Success |", "| --- | --- | --- | --- | ---: |"])
    for row in sorted(combined_rows, key=lambda item: (str(item["candidate"]), str(item["lane"]), int(item["seed"]), str(item["label"]))):
        if str(row["candidate"]) not in (str(campaign["current_canonical_name"]),) + CHALLENGER_ORDER:
            continue
        if str(row["label"]) not in CURRENT_LABELS:
            continue
        lines.append(
            f"| `{row['candidate']}` | {row['lane']} | {row['seed']} | {DISPLAY_NAMES[str(row['label'])]} | {_format_float(_float(row['final_greedy_success']))} |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, combined_rows)
    if json_output is not None:
        _write_json(
            json_output,
            {
                "rows": combined_rows,
                "incumbent_summary": incumbent_summary,
                "challenger_summaries": challenger_summaries,
                "selected_challenger": selected_challenger,
                "promote": promote,
            },
        )


def _render_stage3(
    campaign: dict[str, Any],
    hard_route_csv: Path,
    healthy_route_csv: Path,
    output: Path,
    csv_output: Path | None = None,
    json_output: Path | None = None,
) -> None:
    stage2 = _read_json(Path(campaign["reports"]["stage2_json"]))
    selected = str(stage2["selected_challenger"])
    hard_summary = _route_summary(_read_csv_rows(hard_route_csv))
    healthy_summary = _route_summary(_read_csv_rows(healthy_route_csv))
    hard_pass = _route_pass(campaign, hard_summary)
    healthy_pass = _route_pass(campaign, healthy_summary)
    stability_cases = campaign["stability_cases"]
    challenger_stability = {
        "hard": _stability_summary(
            _stage1_candidate_run_dir(campaign, selected, str(stability_cases["hard"]["lane"]), _int(stability_cases["hard"]["seed"]))
        ),
        "healthy": _stability_summary(
            _stage1_candidate_run_dir(campaign, selected, str(stability_cases["healthy"]["lane"]), _int(stability_cases["healthy"]["seed"]))
        ),
    }
    incumbent_stability = {
        "hard": _stability_summary(
            _current_round6_run_dir(campaign, str(stability_cases["hard"]["lane"]), _int(stability_cases["hard"]["seed"]))
        ),
        "healthy": _stability_summary(
            _current_round6_run_dir(campaign, str(stability_cases["healthy"]["lane"]), _int(stability_cases["healthy"]["seed"]))
        ),
    }
    stability_pass = all(
        summary["stability_class"] != "narrow_spike"
        for summary in (challenger_stability["hard"], challenger_stability["healthy"])
    )
    lines = [
        "# Successor Carryover Challenge Stage 3 Validation",
        "",
        f"- selected challenger: `{selected}`",
        f"- hard route status: `{'pass' if hard_pass else 'fail'}`",
        f"- healthy route status: `{'pass' if healthy_pass else 'fail'}`",
        f"- stability status: `{'pass' if stability_pass else 'fail'}`",
        "",
        "## Route Probes",
        "",
        "| Case | Lane | Seed | Baseline | Fixed-Router Drop | Randomization Drop | Worst-Ablation Drop | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        f"| hard | {hard_summary['lane']} | {hard_summary['seed']} | `{hard_summary['baseline_success']:.4f}` | `{hard_summary['fixed_router_drop']:.4f}` | `{hard_summary['route_randomization_drop']:.4f}` | `{hard_summary['worst_ablation_drop']:.4f}` | `{'pass' if hard_pass else 'fail'}` |",
        f"| healthy | {healthy_summary['lane']} | {healthy_summary['seed']} | `{healthy_summary['baseline_success']:.4f}` | `{healthy_summary['fixed_router_drop']:.4f}` | `{healthy_summary['route_randomization_drop']:.4f}` | `{healthy_summary['worst_ablation_drop']:.4f}` | `{'pass' if healthy_pass else 'fail'}` |",
        "",
        "## Stability",
        "",
        "| Line | Case | Best Round | Best Greedy | Final Greedy | Stability |",
        "| --- | --- | ---: | ---: | ---: | --- |",
        f"| `{campaign['current_canonical_name']}` | hard | `{incumbent_stability['hard']['best_round_index']}` | `{incumbent_stability['hard']['best_round_greedy_success']:.4f}` | `{incumbent_stability['hard']['final_greedy_success']:.4f}` | `{incumbent_stability['hard']['stability_class']}` |",
        f"| `{selected}` | hard | `{challenger_stability['hard']['best_round_index']}` | `{challenger_stability['hard']['best_round_greedy_success']:.4f}` | `{challenger_stability['hard']['final_greedy_success']:.4f}` | `{challenger_stability['hard']['stability_class']}` |",
        f"| `{campaign['current_canonical_name']}` | healthy | `{incumbent_stability['healthy']['best_round_index']}` | `{incumbent_stability['healthy']['best_round_greedy_success']:.4f}` | `{incumbent_stability['healthy']['final_greedy_success']:.4f}` | `{incumbent_stability['healthy']['stability_class']}` |",
        f"| `{selected}` | healthy | `{challenger_stability['healthy']['best_round_index']}` | `{challenger_stability['healthy']['best_round_greedy_success']:.4f}` | `{challenger_stability['healthy']['final_greedy_success']:.4f}` | `{challenger_stability['healthy']['stability_class']}` |",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(
            csv_output,
            [
                {"case": "hard", **hard_summary, "route_pass": hard_pass},
                {"case": "healthy", **healthy_summary, "route_pass": healthy_pass},
            ],
        )
    if json_output is not None:
        _write_json(
            json_output,
            {
                "selected_challenger": selected,
                "hard_route": hard_summary,
                "healthy_route": healthy_summary,
                "route_pass": hard_pass and healthy_pass,
                "stability_pass": stability_pass,
                "challenger_stability": challenger_stability,
                "incumbent_stability": incumbent_stability,
            },
        )


def _render_decision(campaign: dict[str, Any], output: Path) -> None:
    stage1 = _read_json(Path(campaign["reports"]["stage1_json"]))
    stage2 = _read_json(Path(campaign["reports"]["stage2_json"]))
    stage3 = _read_json(Path(campaign["reports"]["stage3_json"]))
    selected = str(stage2["selected_challenger"])
    promote = bool(stage2["promote"])
    route_pass = bool(stage3["route_pass"])
    stability_pass = bool(stage3["stability_pass"])
    if promote and route_pass and stability_pass:
        status = f"{selected} merits a fuller canonization challenge against round6"
    elif route_pass and stability_pass:
        status = "round6 remains the stronger incumbent after carryover challenge"
    else:
        status = "carryover challenge weakens the extra-round challenger lane"
    incumbent = stage2["incumbent_summary"]
    challenger = stage2["challenger_summaries"][selected]
    lines = [
        "# Successor Carryover Challenge Decision Memo",
        "",
        f"- current canonical successor: `{campaign['current_canonical_name']}`",
        f"- selected challenger: `{selected}`",
        f"- promotion status: `{'promote' if promote else 'keep_round6'}`",
        f"- route status: `{'pass' if route_pass else 'fail'}`",
        f"- stability status: `{'pass' if stability_pass else 'fail'}`",
        f"- final status: `{status}`",
        "",
        "## Key Results",
        "",
        f"- round6 hard combined SARE/token/single: `{incumbent['hard_combined_sare']:.4f}` / `{incumbent['hard_combined_token']:.4f}` / `{incumbent['hard_combined_single']:.4f}`",
        f"- {selected} hard combined SARE/token/single: `{challenger['hard_combined_sare']:.4f}` / `{challenger['hard_combined_token']:.4f}` / `{challenger['hard_combined_single']:.4f}`",
        f"- round6 healthy SARE mean: `{stage1['candidate_summaries'][campaign['current_canonical_name']]['healthy_mean']:.4f}`",
        f"- {selected} healthy SARE mean: `{stage1['candidate_summaries'][selected]['healthy_mean']:.4f}`",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Carryover challenger validation for the DoorKey round6 successor")
    sub = parser.add_subparsers(dest="command", required=True)

    registration = sub.add_parser("registration")
    registration.add_argument("--campaign-config", required=True)
    registration.add_argument("--output", required=True)

    stage1 = sub.add_parser("stage1-report")
    stage1.add_argument("--campaign-config", required=True)
    stage1.add_argument("--output", required=True)
    stage1.add_argument("--csv", required=False)
    stage1.add_argument("--json", required=False)

    stage2 = sub.add_parser("stage2-report")
    stage2.add_argument("--campaign-config", required=True)
    stage2.add_argument("--output", required=True)
    stage2.add_argument("--csv", required=False)
    stage2.add_argument("--json", required=False)

    stage3 = sub.add_parser("stage3-report")
    stage3.add_argument("--campaign-config", required=True)
    stage3.add_argument("--hard-route-csv", required=True)
    stage3.add_argument("--healthy-route-csv", required=True)
    stage3.add_argument("--output", required=True)
    stage3.add_argument("--csv", required=False)
    stage3.add_argument("--json", required=False)

    decision = sub.add_parser("decision-memo")
    decision.add_argument("--campaign-config", required=True)
    decision.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    campaign = _load_yaml(Path(args.campaign_config))

    if args.command == "registration":
        _render_registration(campaign, Path(args.output))
        return
    if args.command == "stage1-report":
        _render_stage1(
            campaign,
            Path(args.output),
            csv_output=Path(args.csv) if args.csv else None,
            json_output=Path(args.json) if args.json else None,
        )
        return
    if args.command == "stage2-report":
        _render_stage2(
            campaign,
            Path(args.output),
            csv_output=Path(args.csv) if args.csv else None,
            json_output=Path(args.json) if args.json else None,
        )
        return
    if args.command == "stage3-report":
        _render_stage3(
            campaign,
            Path(args.hard_route_csv),
            Path(args.healthy_route_csv),
            Path(args.output),
            csv_output=Path(args.csv) if args.csv else None,
            json_output=Path(args.json) if args.json else None,
        )
        return
    if args.command == "decision-memo":
        _render_decision(campaign, Path(args.output))
        return
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
