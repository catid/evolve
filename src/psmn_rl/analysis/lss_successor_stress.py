from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_hard_family_saturation import _stability_class
from psmn_rl.analysis.lss_hard_family_saturation import _load_train_run_summary
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


SCREENING_LABELS: tuple[str, ...] = ("recovered_token_dense", "baseline_sare", "kl_lss_sare")


@dataclass(slots=True)
class RunRecord:
    candidate: str
    lane: str
    seed: int
    label: str
    run_dir: Path
    summary: dict[str, Any]


def _discover_runs(root: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    if not root.exists():
        return records
    for summary_path in sorted(root.glob("*/*/seed_*/kl_lss_*/summary.json")):
        run_dir = summary_path.parent
        seed_dir = run_dir.parent
        lane_dir = seed_dir.parent
        candidate_dir = lane_dir.parent
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
        "best_round_route_entropy": _float(best_round.get("collection/route_entropy")),
        "best_round_path_entropy": _float(best_round.get("collection/path_entropy")),
        "final_round_disagreement": _float(final_round.get("collection/disagreement_rate")),
        "final_round_post_unlock_frac": _float(final_round.get("collection/phase_frac_post_unlock")),
    }


def _screening_rows(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [_candidate_row(record) for record in _discover_runs(Path(campaign["stage_roots"]["stage1_screening"]))]
    baseline_root = Path(campaign["stage_roots"]["stage0_blocks"])
    for block in campaign["prospective_blocks"]["blocks"]:
        lane = str(block["lane"])
        for seed in block["seeds"]:
            seed_root = baseline_root / lane / f"seed_{seed}"
            recovered = _load_train_run_summary(seed_root / "token_dense_ent1e3")
            baseline_sare = _load_train_run_summary(seed_root / "sare_ent1e3")
            rows.append(
                {
                    "candidate": "baseline",
                    "lane": lane,
                    "seed": int(seed),
                    "label": "recovered_token_dense",
                    "run_dir": str(seed_root / "token_dense_ent1e3"),
                    "final_greedy_success": _float(recovered.get("eval_greedy_success")),
                }
            )
            rows.append(
                {
                    "candidate": "baseline",
                    "lane": lane,
                    "seed": int(seed),
                    "label": "baseline_sare",
                    "run_dir": str(seed_root / "sare_ent1e3"),
                    "final_greedy_success": _float(baseline_sare.get("eval_greedy_success")),
                }
            )
    return rows


def _candidate_summary(rows: list[dict[str, Any]], candidate: str, labels: tuple[str, ...]) -> dict[str, float]:
    candidate_rows = [row for row in rows if str(row["candidate"]) == candidate and str(row["label"]) in labels]
    return _stats([_float(row["final_greedy_success"]) for row in candidate_rows])


def _preferred_order(campaign: dict[str, Any]) -> list[str]:
    return [str(item) for item in campaign["selection"]["preferred_order"]]


def _challenger_order(campaign: dict[str, Any]) -> list[str]:
    return [str(item) for item in campaign["selection"]["challenger_order"]]


def _pick_best_challenger(campaign: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    order = _challenger_order(campaign)
    stats = {
        candidate: _candidate_summary(rows, candidate, ("kl_lss_sare",))
        for candidate in order
        if any(str(row["candidate"]) == candidate and str(row["label"]) == "kl_lss_sare" for row in rows)
    }
    return max(
        stats,
        key=lambda candidate: (
            stats[candidate]["mean"],
            -stats[candidate]["complete_seed_failures"],
            -order.index(candidate),
        ),
    )


def _pick_route_case(stage1_rows: list[dict[str, Any]], selected_candidate: str, baseline_candidate: str) -> dict[str, Any]:
    best_row: dict[str, Any] | None = None
    best_key: tuple[float, float, str, int] | None = None
    for row in stage1_rows:
        if str(row["candidate"]) != selected_candidate or str(row["label"]) != "kl_lss_sare":
            continue
        lane = str(row["lane"])
        seed = int(row["seed"])
        baseline = next(
            (
                _float(other["final_greedy_success"])
                for other in stage1_rows
                if str(other["candidate"]) == baseline_candidate
                and str(other["label"]) == "kl_lss_sare"
                and str(other["lane"]) == lane
                and int(other["seed"]) == seed
            ),
            0.0,
        )
        key = (_float(row["final_greedy_success"]) - baseline, _float(row["final_greedy_success"]), lane, -seed)
        if best_key is None or key > best_key:
            best_key = key
            best_row = row
    assert best_row is not None
    return {"lane": str(best_row["lane"]), "seed": int(best_row["seed"]), "baseline_candidate": baseline_candidate}


def _control_rows(root: Path) -> list[dict[str, Any]]:
    rows = [_candidate_row(record) for record in _discover_runs(root)]
    return [row for row in rows if str(row["label"]) in {"kl_lss_token_dense", "kl_lss_single_expert"}]


def _fairness_rows(stage1_json: Path, stage2_root: Path, candidates: list[str]) -> list[dict[str, Any]]:
    stage1 = _read_json(stage1_json)
    stage1_rows = [
        row
        for row in stage1["rows"]
        if str(row["candidate"]) in candidates and str(row["label"]) == "kl_lss_sare"
    ]
    control_rows = [row for row in _control_rows(stage2_root) if str(row["candidate"]) in candidates]
    return stage1_rows + control_rows


def _fairness_summary(rows: list[dict[str, Any]], candidate: str) -> dict[str, float]:
    sare_values = [_float(row["final_greedy_success"]) for row in rows if str(row["candidate"]) == candidate and str(row["label"]) == "kl_lss_sare"]
    token_values = [_float(row["final_greedy_success"]) for row in rows if str(row["candidate"]) == candidate and str(row["label"]) == "kl_lss_token_dense"]
    single_values = [_float(row["final_greedy_success"]) for row in rows if str(row["candidate"]) == candidate and str(row["label"]) == "kl_lss_single_expert"]
    sare_stats = _stats(sare_values)
    return {
        "sare_mean": sare_stats["mean"],
        "token_mean": _stats(token_values)["mean"],
        "single_mean": _stats(single_values)["mean"],
        "sare_failures": sare_stats["complete_seed_failures"],
    }


def _promote_challenger(campaign: dict[str, Any], round6: dict[str, float], challenger: dict[str, float]) -> bool:
    min_gain = _float(campaign["selection"]["min_mean_gain"])
    min_margin_gain = _float(campaign["selection"]["min_margin_gain"])
    if challenger["sare_mean"] < challenger["token_mean"] or challenger["sare_mean"] < challenger["single_mean"]:
        return False
    if challenger["sare_failures"] > round6["sare_failures"]:
        return False
    mean_gain = challenger["sare_mean"] - round6["sare_mean"]
    round6_margin = round6["sare_mean"] - max(round6["token_mean"], round6["single_mean"])
    challenger_margin = challenger["sare_mean"] - max(challenger["token_mean"], challenger["single_mean"])
    return mean_gain >= min_gain or (mean_gain >= 0.0 and challenger_margin >= round6_margin + min_margin_gain)


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


def _route_pass(campaign: dict[str, Any], route_summary: dict[str, Any]) -> bool:
    return (
        route_summary["fixed_router_drop"] >= _float(campaign["selection"]["route_drop_min"])
        and route_summary["worst_ablation_drop"] >= _float(campaign["selection"]["route_drop_min"])
        and route_summary["route_randomization_drop"] >= _float(campaign["selection"]["route_randomization_drop_min"])
    )


def _summary_stability(summary: dict[str, Any]) -> str:
    values = [_float(round_summary.get("after_greedy_success")) for round_summary in summary.get("rounds", [])]
    return _stability_class(values)


def _render_registration(campaign: dict[str, Any], output: Path) -> None:
    lines = [
        "# Successor Stress Registration",
        "",
        f"- frozen benchmark pack: `{campaign['frozen_pack']}`",
        f"- current canonical successor pack: `{campaign['current_canonical_pack']}`",
        f"- current canonical successor: `{campaign['current_canonical_name']}`",
        f"- predecessor line: `{campaign['incumbent_candidate_name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Prospective Blocks",
        "",
        f"- fresh stress blocks: `{campaign['prospective_blocks']['blocks']}`",
        "",
        "## Candidate Lineup",
        "",
    ]
    for candidate, meta in campaign["candidates"].items():
        lines.append(f"- `{candidate}`: {meta['description']}")
    lines.extend(
        [
            "",
            "## Plan",
            "",
            "- Stage 1: screen the current successor, predecessor, and later-round challengers on two new prospective DoorKey blocks.",
            "- Stage 2: run matched structured controls only for `round6` and the best challenger from Stage 1.",
            "- Stage 3: route- and stability-check the validated line before deciding whether round6 still holds the incumbent role.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_stage1(campaign: dict[str, Any], output: Path, csv_output: Path | None = None, json_output: Path | None = None) -> None:
    rows = _screening_rows(campaign)
    summaries = {
        candidate: _candidate_summary(rows, candidate, ("kl_lss_sare",))
        for candidate in campaign["candidates"]
        if any(str(row["candidate"]) == candidate and str(row["label"]) == "kl_lss_sare" for row in rows)
    }
    best_challenger = _pick_best_challenger(campaign, rows)
    route_case = _pick_route_case(rows, best_challenger, campaign["current_canonical_name"])
    lines = [
        "# Successor Stress Stage 1 Screening",
        "",
        f"- predecessor mean: `{summaries[campaign['incumbent_candidate_name']]['mean']:.4f}`",
        f"- current canonical `{campaign['current_canonical_name']}` mean: `{summaries[campaign['current_canonical_name']]['mean']:.4f}`",
        f"- best challenger: `{best_challenger}`",
        f"- selected prospective route case: `{route_case}`",
        "",
        "| Candidate | Mean Prospective SARE | Complete-Seed Failures |",
        "| --- | ---: | ---: |",
    ]
    for candidate in _preferred_order(campaign):
        if candidate not in summaries:
            continue
        lines.append(
            f"| `{candidate}` | `{summaries[candidate]['mean']:.4f}` | `{int(summaries[candidate]['complete_seed_failures'])}` |"
        )
    lines.extend(["", "| Candidate | Block | Seed | Variant | Greedy Success |", "| --- | --- | --- | --- | ---: |"])
    for row in sorted(rows, key=lambda item: (str(item["candidate"]), str(item["lane"]), int(item["seed"]), str(item["label"]))):
        if str(row["label"]) not in SCREENING_LABELS:
            continue
        variant_name = {
            "recovered_token_dense": "recovered token_dense",
            "baseline_sare": "baseline PPO SARE",
            "kl_lss_sare": "KL learner-state SARE",
        }[str(row["label"])]
        lines.append(
            f"| `{row['candidate']}` | {row['lane']} | {row['seed']} | {variant_name} | {_format_float(_float(row['final_greedy_success']))} |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, rows)
    if json_output is not None:
        _write_json(
            json_output,
            {
                "summaries": summaries,
                "best_challenger": best_challenger,
                "selected_route_case": route_case,
                "rows": rows,
            },
        )


def _render_stage2(campaign: dict[str, Any], stage1_json: Path, stage2_root: Path, output: Path, csv_output: Path | None = None, json_output: Path | None = None) -> None:
    stage1 = _read_json(stage1_json)
    best_challenger = str(stage1["best_challenger"])
    candidates = [campaign["current_canonical_name"], best_challenger]
    rows = _fairness_rows(stage1_json, stage2_root, candidates)
    summaries = {candidate: _fairness_summary(rows, candidate) for candidate in candidates}
    promote = _promote_challenger(campaign, summaries[campaign["current_canonical_name"]], summaries[best_challenger])
    selected_candidate = best_challenger if promote else campaign["current_canonical_name"]
    lines = [
        "# Successor Stress Stage 2 Fairness",
        "",
        f"- best challenger from Stage 1: `{best_challenger}`",
        f"- validated candidate after fairness: `{selected_candidate}`",
        "",
        "| Candidate | Prospective SARE | Prospective token_dense | Prospective single_expert | SARE-token | SARE-single | SARE Failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in candidates:
        summary = summaries[candidate]
        lines.append(
            f"| `{candidate}` | `{summary['sare_mean']:.4f}` | `{summary['token_mean']:.4f}` | `{summary['single_mean']:.4f}` | `{summary['sare_mean'] - summary['token_mean']:.4f}` | `{summary['sare_mean'] - summary['single_mean']:.4f}` | `{int(summary['sare_failures'])}` |"
        )
    lines.extend(["", "| Candidate | Block | Seed | Variant | Greedy Success |", "| --- | --- | --- | --- | ---: |"])
    for row in sorted(rows, key=lambda item: (str(item["candidate"]), str(item["lane"]), int(item["seed"]), str(item["label"]))):
        lines.append(
            f"| `{row['candidate']}` | {row['lane']} | {row['seed']} | {DISPLAY_NAMES[str(row['label'])]} | {_format_float(_float(row['final_greedy_success']))} |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, rows)
    if json_output is not None:
        _write_json(
            json_output,
            {
                "best_challenger": best_challenger,
                "selected_candidate": selected_candidate,
                "promote": promote,
                "summaries": summaries,
                "rows": rows,
            },
        )


def _render_stage3(campaign: dict[str, Any], stage1_json: Path, stage2_json: Path, route_csv: Path, output: Path, csv_output: Path | None = None, json_output: Path | None = None) -> None:
    stage1 = _read_json(stage1_json)
    stage2 = _read_json(stage2_json)
    selected_candidate = str(stage2["selected_candidate"])
    route_rows = _read_csv_rows(route_csv)
    route_summary = _route_summary(route_rows)
    route_ok = _route_pass(campaign, route_summary)
    lane = str(route_summary["lane"])
    seed = int(route_summary["seed"])
    selected_summary = _read_json(
        Path(campaign["stage_roots"]["stage1_screening"]) / selected_candidate / lane / f"seed_{seed}" / "kl_lss_sare" / "summary.json"
    )
    selected_stability = _summary_stability(selected_summary)
    selected_stability_ok = selected_stability == "stable_plateau"
    lines = [
        "# Successor Stress Stage 3 Validation",
        "",
        f"- best challenger: `{stage1['best_challenger']}`",
        f"- validated candidate: `{selected_candidate}`",
        f"- route case: `{lane}` seed `{seed}`",
        f"- route status: `{'pass' if route_ok else 'fail'}`",
        f"- selected candidate stability: `{selected_stability}`",
        f"- stability status: `{'pass' if selected_stability_ok else 'fail'}`",
        "",
        f"- fixed-router drop: `{route_summary['fixed_router_drop']:.4f}`",
        f"- route-randomization drop: `{route_summary['route_randomization_drop']:.4f}`",
        f"- worst-ablation drop: `{route_summary['worst_ablation_drop']:.4f}`",
    ]
    if selected_candidate != campaign["current_canonical_name"]:
        round6_summary = _read_json(
            Path(campaign["stage_roots"]["stage1_screening"]) / campaign["current_canonical_name"] / lane / f"seed_{seed}" / "kl_lss_sare" / "summary.json"
        )
        lines.append(f"- round6 stability on same case: `{_summary_stability(round6_summary)}`")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, route_rows)
    if json_output is not None:
        _write_json(
            json_output,
            {
                "best_challenger": stage1["best_challenger"],
                "selected_candidate": selected_candidate,
                "route_summary": route_summary,
                "route_pass": route_ok,
                "selected_stability": selected_stability,
                "stability_pass": selected_stability_ok,
            },
        )


def _render_decision(campaign: dict[str, Any], stage1_json: Path, stage2_json: Path, stage3_json: Path, output: Path) -> None:
    stage1 = _read_json(stage1_json)
    stage2 = _read_json(stage2_json)
    stage3 = _read_json(stage3_json)
    round6 = stage2["summaries"][campaign["current_canonical_name"]]
    challenger = stage2["summaries"][stage2["best_challenger"]]
    selected_candidate = str(stage2["selected_candidate"])
    route_ok = bool(stage3["route_pass"])
    stability_ok = bool(stage3["stability_pass"])
    if selected_candidate != campaign["current_canonical_name"] and route_ok and stability_ok:
        status = f"{selected_candidate} merits a fuller follow-up against round6"
    elif route_ok and stability_ok:
        status = "round6 remains the validated incumbent on new prospective blocks"
    else:
        status = "prospective stress weakens the successor line"
    lines = [
        "# Successor Stress Decision Memo",
        "",
        f"- current canonical successor: `{campaign['current_canonical_name']}`",
        f"- best screened challenger: `{stage1['best_challenger']}`",
        f"- fairness-validated candidate: `{selected_candidate}`",
        f"- route status: `{'pass' if route_ok else 'fail'}`",
        f"- stability status: `{'pass' if stability_ok else 'fail'}`",
        f"- final status: `{status}`",
        "",
        "## Key Results",
        "",
        f"- round6 prospective SARE/token/single means: `{round6['sare_mean']:.4f}` / `{round6['token_mean']:.4f}` / `{round6['single_mean']:.4f}`",
        f"- {stage2['best_challenger']} prospective SARE/token/single means: `{challenger['sare_mean']:.4f}` / `{challenger['token_mean']:.4f}` / `{challenger['single_mean']:.4f}`",
        f"- route case: `{stage3['route_summary']['lane']}` seed `{stage3['route_summary']['seed']}`",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prospective stress validation for the DoorKey round6 successor")
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
    stage2.add_argument("--stage1-json", required=True)
    stage2.add_argument("--stage2-root", required=True)
    stage2.add_argument("--output", required=True)
    stage2.add_argument("--csv", required=False)
    stage2.add_argument("--json", required=False)

    stage3 = sub.add_parser("stage3-report")
    stage3.add_argument("--campaign-config", required=True)
    stage3.add_argument("--stage1-json", required=True)
    stage3.add_argument("--stage2-json", required=True)
    stage3.add_argument("--route-csv", required=True)
    stage3.add_argument("--output", required=True)
    stage3.add_argument("--csv", required=False)
    stage3.add_argument("--json", required=False)

    decision = sub.add_parser("decision-memo")
    decision.add_argument("--campaign-config", required=True)
    decision.add_argument("--stage1-json", required=True)
    decision.add_argument("--stage2-json", required=True)
    decision.add_argument("--stage3-json", required=True)
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
            Path(args.stage1_json),
            Path(args.stage2_root),
            Path(args.output),
            csv_output=Path(args.csv) if args.csv else None,
            json_output=Path(args.json) if args.json else None,
        )
        return
    if args.command == "stage3-report":
        _render_stage3(
            campaign,
            Path(args.stage1_json),
            Path(args.stage2_json),
            Path(args.route_csv),
            Path(args.output),
            csv_output=Path(args.csv) if args.csv else None,
            json_output=Path(args.json) if args.json else None,
        )
        return
    if args.command == "decision-memo":
        _render_decision(
            campaign,
            Path(args.stage1_json),
            Path(args.stage2_json),
            Path(args.stage3_json),
            Path(args.output),
        )
        return
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
