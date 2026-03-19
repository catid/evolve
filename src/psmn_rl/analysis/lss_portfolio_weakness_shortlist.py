from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_post_pass_campaign import _load_yaml, _read_json, _write_json
from psmn_rl.utils.io import get_git_commit, get_git_dirty


def _aggregate(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_candidate: dict[str, dict[str, Any]] = {}
    for entry in entries:
        candidate = str(entry["candidate"])
        item = by_candidate.setdefault(
            candidate,
            {
                "candidate": candidate,
                "sources": [],
                "best_seed_269": float(entry["seed_269"]),
                "best_lane_mean": float(entry["lane_mean"]),
                "min_seed_277": float(entry["seed_277"]) if entry["seed_277"] is not None else None,
            },
        )
        item["sources"].append(str(entry["source"]))
        item["best_seed_269"] = max(float(item["best_seed_269"]), float(entry["seed_269"]))
        item["best_lane_mean"] = max(float(item["best_lane_mean"]), float(entry["lane_mean"]))
        seed_277 = float(entry["seed_277"]) if entry["seed_277"] is not None else None
        if seed_277 is not None:
            item["min_seed_277"] = seed_277 if item["min_seed_277"] is None else min(float(item["min_seed_277"]), seed_277)
    for item in by_candidate.values():
        item["sources"] = sorted(set(item["sources"]))
        item["source_count"] = len(item["sources"])
        item["bucket"] = _bucket(item)
    return sorted(
        by_candidate.values(),
        key=lambda item: (-float(item["best_seed_269"]), -float(item["best_lane_mean"]), -int(item["source_count"]), str(item["candidate"])),
    )


def _bucket(item: dict[str, Any]) -> str:
    best_seed_269 = float(item["best_seed_269"])
    best_lane_mean = float(item["best_lane_mean"])
    candidate = str(item["candidate"])
    if best_seed_269 >= 1.0 - 1e-9 and best_lane_mean >= 1.0 - 1e-9:
        if candidate.startswith("round"):
            return "conservative_round_prior"
        return "structural_full_lane_prior"
    if best_seed_269 >= 1.0 - 1e-9 and best_lane_mean >= 0.95:
        return "partial_revisit"
    if best_seed_269 >= 1.0 - 1e-9:
        return "local_only_fix"
    if best_seed_269 >= 0.984375 - 1e-9 and best_lane_mean >= 0.99:
        return "incumbent_parity"
    return "dead_end"


def render_report(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    leaderboard = _read_json(Path("outputs/reports/portfolio_weakness_leaderboard.json"))
    aggregated = _aggregate(list(leaderboard["entries"]))

    buckets: dict[str, list[str]] = {}
    for item in aggregated:
        buckets.setdefault(str(item["bucket"]), []).append(str(item["candidate"]))

    conservative = buckets.get("conservative_round_prior", [])
    structural = buckets.get("structural_full_lane_prior", [])
    partial = buckets.get("partial_revisit", [])
    local_only = buckets.get("local_only_fix", [])
    dead = buckets.get("dead_end", [])

    lines = [
        "# Portfolio Weakness Shortlist",
        "",
        f"- source campaign: `{campaign['name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        f"- bounded weakness seed: `{leaderboard['active_bounded_weakness']['lane']}/{leaderboard['active_bounded_weakness']['seed']}`",
        "",
        "## Aggregated Historical Priors",
        "",
        "| Candidate | Best Seed 269 | Best prospective_h Mean | Min Seed 277 | Sources | Bucket |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for item in aggregated:
        min_seed_277 = "n/a" if item["min_seed_277"] is None else f"{float(item['min_seed_277']):.4f}"
        lines.append(
            f"| `{item['candidate']}` | `{float(item['best_seed_269']):.4f}` | `{float(item['best_lane_mean']):.4f}` | `{min_seed_277}` | `{item['sources']}` | `{item['bucket']}` |"
        )

    lines.extend(
        [
            "",
            "## Search Prior",
            "",
            f"- conservative round-count priors: `{conservative}`",
            f"- structural full-lane priors: `{structural}`",
            f"- partial revisits worth considering only after the full-lane priors: `{partial}`",
            f"- local-only fixes to deprioritize: `{local_only}`",
            f"- clear dead ends on the weakness surface: `{dead}`",
            "",
            "## Interpretation",
            "",
            "- The cleanest bounded-search priors are the simple round-count families that solve `prospective_h/269` and keep the whole `prospective_h` lane at `1.0000` without additional mechanism complexity.",
            "- Structural one-offs such as `door3_post5` also solve the lane cleanly, but they are less conservative starting points because they add a different mechanism family and still failed broader league selection.",
            "- Local-only fixes that solve `269` while degrading `277` should not be treated as first-line priors; they are useful only if the conservative full-lane priors are exhausted.",
            "- Future challenger work can now start from a narrow evidence-backed shortlist instead of replaying the entire historical tie bucket.",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "aggregated": aggregated,
                "buckets": buckets,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a bounded-search shortlist from the portfolio weakness leaderboard")
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
