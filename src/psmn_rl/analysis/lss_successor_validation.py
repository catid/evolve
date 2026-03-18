from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


CURRENT_CONTROLS: tuple[str, ...] = ("kl_lss_sare", "kl_lss_token_dense", "kl_lss_single_expert")
SCREENING_LABELS: tuple[str, ...] = ("recovered_token_dense", "baseline_sare", "kl_lss_sare")
PREFERRED_ORDER: tuple[str, ...] = ("round6", "round7", "round5", "post_unlock_weighted")


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


def _candidate_display_name(candidate: str) -> str:
    return candidate.replace("_", " ")


def _screening_rows(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [_candidate_row(record) for record in _discover_runs(Path(campaign["stage_roots"]["stage1_screening"]))]
    baseline_root = Path(campaign["stage_roots"]["stage0_blocks"])
    for block in campaign["fresh_blocks"]["blocks"]:
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


def _candidate_block_summary(rows: list[dict[str, Any]], candidate: str) -> list[dict[str, Any]]:
    block_rows: list[dict[str, Any]] = []
    lanes = sorted({str(row["lane"]) for row in rows if str(row["candidate"]) == candidate and str(row["label"]) == "kl_lss_sare"})
    for lane in lanes:
        lane_values = [
            _float(row["final_greedy_success"])
            for row in rows
            if str(row["candidate"]) == candidate and str(row["lane"]) == lane and str(row["label"]) == "kl_lss_sare"
        ]
        block_rows.append({"candidate": candidate, "lane": lane, "sare_mean": _stats(lane_values)["mean"]})
    return block_rows


def _pick_stage1_best(rows: list[dict[str, Any]]) -> str:
    candidate_stats = {
        candidate: _candidate_summary(rows, candidate, ("kl_lss_sare",))
        for candidate in PREFERRED_ORDER
        if any(str(row["candidate"]) == candidate and str(row["label"]) == "kl_lss_sare" for row in rows)
    }
    return max(
        candidate_stats,
        key=lambda candidate: (
            candidate_stats[candidate]["mean"],
            -candidate_stats[candidate]["complete_seed_failures"],
            -PREFERRED_ORDER.index(candidate),
        ),
    )


def _pick_route_case(stage1_rows: list[dict[str, Any]], selected_candidate: str) -> dict[str, Any]:
    best_row: dict[str, Any] | None = None
    best_key: tuple[float, float, str, int] | None = None
    for row in stage1_rows:
        if str(row["candidate"]) != selected_candidate or str(row["label"]) != "kl_lss_sare":
            continue
        lane = str(row["lane"])
        seed = int(row["seed"])
        incumbent = next(
            (
                _float(other["final_greedy_success"])
                for other in stage1_rows
                if str(other["candidate"]) == "post_unlock_weighted"
                and str(other["label"]) == "kl_lss_sare"
                and str(other["lane"]) == lane
                and int(other["seed"]) == seed
            ),
            0.0,
        )
        key = (_float(row["final_greedy_success"]) - incumbent, _float(row["final_greedy_success"]), lane, -seed)
        if best_key is None or key > best_key:
            best_key = key
            best_row = row
    assert best_row is not None
    return {"lane": str(best_row["lane"]), "seed": int(best_row["seed"])}


def _fairness_rows(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    return [_candidate_row(record) for record in _discover_runs(Path(campaign["stage_roots"]["stage2_fairness"]))]


def _fairness_summary(rows: list[dict[str, Any]], candidate: str) -> dict[str, float]:
    return {
        "sare_mean": _stats(
            [_float(row["final_greedy_success"]) for row in rows if str(row["candidate"]) == candidate and str(row["label"]) == "kl_lss_sare"]
        )["mean"],
        "token_mean": _stats(
            [_float(row["final_greedy_success"]) for row in rows if str(row["candidate"]) == candidate and str(row["label"]) == "kl_lss_token_dense"]
        )["mean"],
        "single_mean": _stats(
            [_float(row["final_greedy_success"]) for row in rows if str(row["candidate"]) == candidate and str(row["label"]) == "kl_lss_single_expert"]
        )["mean"],
        "sare_failures": _stats(
            [_float(row["final_greedy_success"]) for row in rows if str(row["candidate"]) == candidate and str(row["label"]) == "kl_lss_sare"]
        )["complete_seed_failures"],
    }


def _select_validation_candidate(campaign: dict[str, Any], summaries: dict[str, dict[str, float]]) -> str:
    round6 = summaries["round6"]
    round7 = summaries["round7"]
    gain = round7["sare_mean"] - round6["sare_mean"]
    if (
        gain >= _float(campaign["selection"]["round7_min_gain"])
        and round7["sare_mean"] >= round7["token_mean"]
        and round7["sare_mean"] >= round7["single_mean"]
    ):
        return "round7"
    return "round6"


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


def _render_registration(campaign: dict[str, Any], output: Path) -> None:
    lines = [
        "# Successor Validation Registration",
        "",
        f"- frozen benchmark pack: `{campaign['frozen_pack']}`",
        f"- current canonical successor pack: `{campaign['current_canonical_pack']}`",
        f"- current canonical successor: `{campaign['current_canonical_name']}`",
        f"- predecessor line: `{campaign['incumbent_candidate_name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Fresh Blocks",
        "",
        f"- fresh validation blocks: `{campaign['fresh_blocks']['blocks']}`",
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
            "- Stage 1: screen the successor line on genuinely fresh DoorKey blocks against the thaw-qualified predecessor.",
            "- Stage 2: run matched structured controls for `round6` and `round7` on the same blocks.",
            "- Stage 3: run one fresh route probe for the best validated line and write the decision memo.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_stage1(campaign: dict[str, Any], output: Path, csv_output: Path | None = None, json_output: Path | None = None) -> None:
    rows = _screening_rows(campaign)
    summaries = {candidate: _candidate_summary(rows, candidate, ("kl_lss_sare",)) for candidate in campaign["candidates"]}
    selected_candidate = _pick_stage1_best(rows)
    route_case = _pick_route_case(rows, selected_candidate)
    lines = [
        "# Successor Validation Stage 1 Screening",
        "",
        f"- incumbent predecessor mean: `{summaries['post_unlock_weighted']['mean']:.4f}`",
        f"- round5 mean: `{summaries['round5']['mean']:.4f}`",
        f"- round6 mean: `{summaries['round6']['mean']:.4f}`",
        f"- round7 mean: `{summaries['round7']['mean']:.4f}`",
        f"- selected fresh route case: `{route_case}`",
        "",
        "| Candidate | Mean Fresh SARE | Complete-Seed Failures |",
        "| --- | ---: | ---: |",
    ]
    for candidate in PREFERRED_ORDER:
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
                "selected_candidate": selected_candidate,
                "selected_route_case": route_case,
                "rows": rows,
            },
        )


def _render_stage2(campaign: dict[str, Any], output: Path, csv_output: Path | None = None, json_output: Path | None = None) -> None:
    rows = _fairness_rows(campaign)
    summaries = {candidate: _fairness_summary(rows, candidate) for candidate in campaign["fairness_candidates"]}
    selected_candidate = _select_validation_candidate(campaign, summaries)
    lines = [
        "# Successor Validation Stage 2 Fairness",
        "",
        f"- selected validated candidate: `{selected_candidate}`",
        "",
        "| Candidate | Fresh SARE | Fresh token_dense | Fresh single_expert | SARE-token | SARE-single | SARE Failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in campaign["fairness_candidates"]:
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
        _write_json(json_output, {"summaries": summaries, "selected_candidate": selected_candidate, "rows": rows})


def _render_route(campaign: dict[str, Any], route_csv: Path, output: Path, csv_output: Path | None = None, json_output: Path | None = None) -> None:
    rows = _read_csv_rows(route_csv)
    summary = _route_summary(rows)
    passed = _route_pass(campaign, summary)
    lines = [
        "# Successor Validation Stage 3 Route",
        "",
        f"- lane: `{summary['lane']}`",
        f"- seed: `{summary['seed']}`",
        f"- baseline greedy success: `{summary['baseline_success']:.4f}`",
        f"- fixed-router drop: `{summary['fixed_router_drop']:.4f}`",
        f"- route-randomization drop: `{summary['route_randomization_drop']:.4f}`",
        f"- worst-ablation drop: `{summary['worst_ablation_drop']:.4f}`",
        f"- route status: `{'pass' if passed else 'fail'}`",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, rows)
    if json_output is not None:
        _write_json(json_output, {"summary": summary, "route_pass": passed, "rows": rows})


def _render_decision(campaign: dict[str, Any], stage1_json: Path, stage2_json: Path, route_json: Path, output: Path) -> None:
    stage1 = _read_json(stage1_json)
    stage2 = _read_json(stage2_json)
    route = _read_json(route_json)
    selected = str(stage2["selected_candidate"])
    round6 = stage2["summaries"]["round6"]
    round7 = stage2["summaries"]["round7"]
    route_pass = bool(route["route_pass"])
    if selected == "round7" and route_pass:
        status = "round7 looks promising but remains exploratory"
    elif route_pass and round6["sare_mean"] >= round6["token_mean"]:
        status = "round6 remains the best validated successor on fresh blocks"
    else:
        status = "fresh validation weakens the successor line"
    lines = [
        "# Successor Validation Decision Memo",
        "",
        f"- current canonical successor: `{campaign['current_canonical_name']}`",
        f"- best fresh-screened candidate: `{stage1['selected_candidate']}`",
        f"- best fairness-validated candidate: `{selected}`",
        f"- route status: `{'pass' if route_pass else 'fail'}`",
        f"- final status: `{status}`",
        "",
        "## Key Results",
        "",
        f"- round6 fresh SARE/token/single means: `{round6['sare_mean']:.4f}` / `{round6['token_mean']:.4f}` / `{round6['single_mean']:.4f}`",
        f"- round7 fresh SARE/token/single means: `{round7['sare_mean']:.4f}` / `{round7['token_mean']:.4f}` / `{round7['single_mean']:.4f}`",
        f"- route case: `{route['summary']['lane']}` seed `{route['summary']['seed']}`",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prospective validation for the DoorKey round6 successor")
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

    route = sub.add_parser("route-report")
    route.add_argument("--campaign-config", required=True)
    route.add_argument("--route-csv", required=True)
    route.add_argument("--output", required=True)
    route.add_argument("--csv", required=False)
    route.add_argument("--json", required=False)

    decision = sub.add_parser("decision-memo")
    decision.add_argument("--campaign-config", required=True)
    decision.add_argument("--stage1-json", required=True)
    decision.add_argument("--stage2-json", required=True)
    decision.add_argument("--route-json", required=True)
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
    if args.command == "route-report":
        _render_route(
            campaign,
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
            Path(args.route_json),
            Path(args.output),
        )
        return
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
