from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_portfolio_signal_atlas import _classify_seed
from psmn_rl.analysis.lss_post_pass_campaign import _float, _load_yaml, _read_json, _write_json
from psmn_rl.utils.io import get_git_commit, get_git_dirty

REPORT_SOURCES = {
    "stress_extended": "outputs/reports/successor_stress_extended_stage1_screening.json",
    "migration": "outputs/reports/successor_migration_stage1_screening.json",
}

SOURCE_NOTES = {
    "stress_extended": "the broader campaign did not yield a meaningful post-control replacement",
    "migration": "the broader campaign still failed to produce a migration-quality challenger",
}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path)
    rows: list[dict[str, Any]] = []
    for key in ("detail_rows", "rows", "screen_rows", "candidate_rows"):
        value = data.get(key)
        if isinstance(value, list):
            rows.extend(value)
    return rows


def _seed_row(rows: list[dict[str, Any]], candidate: str, label: str, lane: str, seed: int) -> dict[str, Any] | None:
    for row in rows:
        if (
            str(row.get("candidate")) == candidate
            and str(row.get("label")) == label
            and str(row.get("lane")) == lane
            and int(row.get("seed", -1)) == seed
        ):
            return row
    return None


def _candidate_entry(source: str, rows: list[dict[str, Any]], candidate: str) -> dict[str, Any] | None:
    seed_269 = _seed_row(rows, candidate, "kl_lss_sare", "prospective_h", 269)
    if seed_269 is None:
        return None
    seed_271 = _seed_row(rows, candidate, "kl_lss_sare", "prospective_h", 271)
    seed_277 = _seed_row(rows, candidate, "kl_lss_sare", "prospective_h", 277)
    return {
        "source": source,
        "candidate": candidate,
        "seed_269": _float(seed_269["final_greedy_success"]),
        "seed_271": _float(seed_271["final_greedy_success"]) if seed_271 is not None else None,
        "seed_277": _float(seed_277["final_greedy_success"]) if seed_277 is not None else None,
        "lane_mean": _float(seed_269["final_greedy_success"])
        + (_float(seed_271["final_greedy_success"]) if seed_271 is not None else 0.0)
        + (_float(seed_277["final_greedy_success"]) if seed_277 is not None else 0.0),
        "campaign_note": SOURCE_NOTES[source],
    }


def _active_controls(campaign: dict[str, Any]) -> dict[str, Any]:
    lane = "prospective_h"
    seed = 269
    sare_root = Path(campaign["current_round6_sare_roots"][lane])
    ctrl_root = Path(campaign["current_round6_control_roots"][lane])
    def _summary(root: Path, variant: str) -> dict[str, Any]:
        return _read_json(root / f"seed_{seed}" / variant / "summary.json")

    sare = _summary(sare_root, "kl_lss_sare")
    token = _summary(ctrl_root, "kl_lss_token_dense")
    single = _summary(ctrl_root, "kl_lss_single_expert")
    return {
        "lane": lane,
        "seed": seed,
        "round6": _float(sare["final_greedy_success"]),
        "token_dense": _float(token["final_greedy_success"]),
        "single_expert": _float(single["final_greedy_success"]),
        "classification": _classify_seed(
            _float(sare["final_greedy_success"]),
            _float(token["final_greedy_success"]),
            _float(single["final_greedy_success"]),
        ),
    }


def render_report(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    active = _active_controls(campaign)
    entries: list[dict[str, Any]] = []
    for source, report in REPORT_SOURCES.items():
        rows = _load_rows(Path(report))
        candidates = sorted({str(row.get("candidate")) for row in rows if row.get("candidate") is not None})
        for candidate in candidates:
            entry = _candidate_entry(source, rows, candidate)
            if entry is not None:
                entry["lane_mean"] = entry["lane_mean"] / 3.0
                entries.append(entry)

    entries.sort(key=lambda item: (-float(item["seed_269"]), -float(item["lane_mean"]), item["candidate"], item["source"]))
    full_solvers = [item for item in entries if float(item["seed_269"]) >= 1.0 - 1e-9]
    full_solver_names = sorted({str(item["candidate"]) for item in full_solvers})

    lines = [
        "# Portfolio Weakness Seed Leaderboard",
        "",
        f"- source campaign: `{campaign['name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        f"- bounded weakness seed: `{active['lane']}/{active['seed']}`",
        f"- active benchmark/control state: `round6={active['round6']:.4f}`, `token_dense={active['token_dense']:.4f}`, `single_expert={active['single_expert']:.4f}`, `classification={active['classification']}`",
        "",
        "## Historical Leaderboard On The Weakness Seed",
        "",
        "| Source | Candidate | Seed 269 | Seed 271 | Seed 277 | prospective_h Mean | Note |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for entry in entries:
        seed_271 = "n/a" if entry["seed_271"] is None else f"{float(entry['seed_271']):.4f}"
        seed_277 = "n/a" if entry["seed_277"] is None else f"{float(entry['seed_277']):.4f}"
        lines.append(
            f"| `{entry['source']}` | `{entry['candidate']}` | `{float(entry['seed_269']):.4f}` | `{seed_271}` | `{seed_277}` | `{float(entry['lane_mean']):.4f}` | {entry['campaign_note']} |"
        )

    lines.extend(
        [
            "",
            "## Full Solvers Of The Weakness Seed",
            "",
            f"- candidates with `seed_269 = 1.0000`: `{full_solver_names}`",
            f"- unique count: `{len(full_solver_names)}`",
            "",
            "## Interpretation",
            "",
            "- The bounded weakness is real but not unique to `round6`: several historical challengers already solve `prospective_h/269` cleanly.",
            "- That makes `269` a useful target seed for future search, but not a sufficient promotion criterion by itself. The repo history already shows that local fixes here can still fail the broader post-control league.",
            "- The right next-step use of this leaderboard is to bias future bounded search toward families that solve `269` without spending more budget on families that either stay at `0.9844` or collapse well below baseline.",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "active_bounded_weakness": active,
                "entries": entries,
                "full_solvers": full_solvers,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a historical leaderboard for the bounded portfolio weakness seed")
    parser.add_argument("--campaign-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json", required=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    campaign = _load_yaml(Path(args.campaign_config))
    render_report(campaign, Path(args.output), Path(args.json) if args.json else None)


if __name__ == "__main__":
    main()
