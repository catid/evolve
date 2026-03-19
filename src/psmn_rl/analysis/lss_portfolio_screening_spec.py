from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_post_pass_campaign import _load_yaml, _read_json, _write_json
from psmn_rl.utils.io import get_git_commit, get_git_dirty

RESTART_POLICY = Path("outputs/reports/portfolio_restart_policy.json")
WEAKNESS_DOSSIER = Path("outputs/reports/portfolio_weakness_dossier.json")
DISCRIMINATOR_REPORT = Path("outputs/reports/portfolio_discriminator_report.json")
SIGNAL_ATLAS = Path("outputs/reports/portfolio_signal_atlas.json")


def _score_rule(support_233: float | None, weakness_269: float, guardrail_277: float, incumbent_269: float) -> str:
    if support_233 is not None and support_233 < 1.0 - 1e-9:
        return "prune_support_regression"
    if guardrail_277 < 1.0 - 1e-9:
        return "prune_guardrail_regression"
    if weakness_269 <= incumbent_269 + 1e-9:
        return "tie_only_no_weakness_gain"
    if support_233 is None:
        return "needs_support_measurement"
    return "advance_for_broader_dev"


def _atlas_row(rows: list[dict[str, Any]], lane: str, seed: int) -> dict[str, Any]:
    for row in rows:
        if str(row["lane"]) == lane and int(row["seed"]) == seed:
            return row
    raise KeyError((lane, seed))


def render_report(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    restart = _read_json(RESTART_POLICY)
    weakness = _read_json(WEAKNESS_DOSSIER)
    discriminator = _read_json(DISCRIMINATOR_REPORT)
    atlas = _read_json(SIGNAL_ATLAS)

    restart_rows = list(restart["rows"])
    atlas_rows = list(atlas["rows"])
    incumbent_row = next(row for row in restart_rows if str(row["candidate"]) == "round6")
    incumbent_269 = float(incumbent_row["weakness_269"])

    primary_weakness = dict(weakness["primary_weakness"])
    discriminator_map = dict(discriminator["classifications"])
    differentiator = [key for key, value in discriminator_map.items() if str(value) == "route_control_differentiator"]
    global_hard = [key for key, value in discriminator_map.items() if str(value) == "global_hard_failure"]

    support_lane, support_seed = "prospective_f", 233
    guardrail_lane, guardrail_seed = "prospective_h", 277
    weakness_lane, weakness_seed = str(primary_weakness["lane"]), int(primary_weakness["seed"])
    sentinel_lane, sentinel_seed = "prospective_c", 193

    support_row = _atlas_row(atlas_rows, support_lane, support_seed)
    weakness_row = _atlas_row(atlas_rows, weakness_lane, weakness_seed)
    guardrail_row = _atlas_row(atlas_rows, guardrail_lane, guardrail_seed)
    sentinel_row = _atlas_row(atlas_rows, sentinel_lane, sentinel_seed)

    candidate_rows: list[dict[str, Any]] = []
    for row in restart_rows:
        if str(row["candidate"]) == "round6":
            continue
        candidate_rows.append(
            {
                "candidate": str(row["candidate"]),
                "policy_bucket": str(row["policy_bucket"]),
                "support_233": float(row["support_233"]) if row["support_233"] is not None else None,
                "weakness_269": float(row["weakness_269"]),
                "guardrail_277": float(row["guardrail_277"]),
                "screen_rule": _score_rule(
                    float(row["support_233"]) if row["support_233"] is not None else None,
                    float(row["weakness_269"]),
                    float(row["guardrail_277"]),
                    incumbent_269,
                ),
            }
        )

    lines = [
        "# Portfolio Screening Spec",
        "",
        f"- source campaign: `{campaign['name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        f"- restart default: `{restart['restart_default']}`",
        f"- reserve priors: `{restart['reserve_priors']}`",
        f"- retired priors: `{restart['retired_priors']}`",
        "",
        "## Minimal Seed Roles",
        "",
        "| Role | Lane | Seed | round6 | token_dense | single_expert | Classification | Use |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        f"| `sentinel` | `{sentinel_lane}` | {sentinel_seed} | `{sentinel_row['sare']:.4f}` | `{sentinel_row['token_dense']:.4f}` | `{sentinel_row['single_expert']:.4f}` | `{sentinel_row['classification']}` | `track only; do not rank on this seed` |",
        f"| `ranking_support` | `{support_lane}` | {support_seed} | `{support_row['sare']:.4f}` | `{support_row['token_dense']:.4f}` | `{support_row['single_expert']:.4f}` | `{support_row['classification']}` | `prune if a candidate regresses below 1.0000` |",
        f"| `ranking_weakness` | `{weakness_lane}` | {weakness_seed} | `{weakness_row['sare']:.4f}` | `{weakness_row['token_dense']:.4f}` | `{weakness_row['single_expert']:.4f}` | `{weakness_row['classification']}` | `require improvement above round6 before broader dev spend` |",
        f"| `guardrail` | `{guardrail_lane}` | {guardrail_seed} | `{guardrail_row['sare']:.4f}` | `{guardrail_row['token_dense']:.4f}` | `{guardrail_row['single_expert']:.4f}` | `{guardrail_row['classification']}` | `prune if a candidate falls below 1.0000` |",
        "",
        "## Restart Frontier Under The Spec",
        "",
        "| Candidate | Policy Bucket | support_233 | weakness_269 | guardrail_277 | Screen Rule |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in candidate_rows:
        support = "n/a" if row["support_233"] is None else f"{row['support_233']:.4f}"
        lines.append(
            f"| `{row['candidate']}` | `{row['policy_bucket']}` | `{support}` | `{row['weakness_269']:.4f}` | `{row['guardrail_277']:.4f}` | `{row['screen_rule']}` |"
        )

    lines.extend(
        [
            "",
            "## Screening Policy",
            "",
            f"- global-hard sentinels already confirmed: `{global_hard}`",
            f"- measured differentiator seeds: `{differentiator}`",
            "- Any bounded mini-sweep should include all four roles above before it gets a broader dev rerun.",
            "- A candidate that regresses on `prospective_f/233` or `prospective_h/277` should be pruned immediately after rerun.",
            "- A candidate that only ties `prospective_h/269` should not advance, because the measured frontier already shows that tie bucket does not survive broader post-control comparison.",
            "- A candidate only earns broader dev/fairness budget if it keeps `f/233 = 1.0000`, keeps `h/277 = 1.0000`, and improves `h/269` above `round6 = 0.9844`.",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "candidate_rows": candidate_rows,
                "restart_default": restart["restart_default"],
                "reserve_priors": restart["reserve_priors"],
                "retired_priors": restart["retired_priors"],
                "seed_roles": {
                    "sentinel": {"lane": sentinel_lane, "seed": sentinel_seed},
                    "ranking_support": {"lane": support_lane, "seed": support_seed},
                    "ranking_weakness": {"lane": weakness_lane, "seed": weakness_seed},
                    "guardrail": {"lane": guardrail_lane, "seed": guardrail_seed},
                },
                "global_hard": global_hard,
                "differentiator": differentiator,
                "incumbent_269": incumbent_269,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an operational screening spec from the measured portfolio frontier")
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
