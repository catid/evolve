from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_post_pass_campaign import _load_yaml, _read_json, _write_json
from psmn_rl.utils.io import get_git_commit, get_git_dirty

RESTART_POLICY = Path("outputs/reports/portfolio_restart_policy.json")
SEED_PACK_SCORER = Path("outputs/reports/portfolio_seed_pack_scorer.json")
FRONTIER_REPLAY = Path("outputs/reports/portfolio_frontier_replay.json")
SEED_PACK = Path("outputs/reports/portfolio_seed_pack.json")


def _manifest_bucket(
    candidate: str,
    scorer_verdict: str | None,
    replay_verdict: str | None,
    restart_default: list[str],
) -> str:
    if candidate == "round6":
        return "active_benchmark"
    if candidate in restart_default:
        return "default_restart_prior"
    if replay_verdict == "advance_for_broader_dev":
        return "replay_validated_alternate"
    if scorer_verdict == "hold_seed_clean_but_below_incumbent":
        return "hold_only_prior"
    if scorer_verdict and scorer_verdict.startswith("prune_"):
        return "retired_prior"
    if scorer_verdict == "advance_for_broader_dev":
        return "seed_clean_unconfirmed_alternate"
    return "unclassified"


def _row_by_candidate(rows: list[dict[str, Any]], candidate_key: str, candidate: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row[candidate_key]) == candidate:
            return row
    return None


def render_report(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    restart = _read_json(RESTART_POLICY)
    scorer = _read_json(SEED_PACK_SCORER)
    replay = _read_json(FRONTIER_REPLAY)
    seed_pack = _read_json(SEED_PACK)

    restart_default = list(restart["restart_default"])
    scorer_rows = list(scorer["rows"])
    replay_rows = list(replay["rows"])

    candidates = [
        "round6",
        *restart_default,
        *[str(row["candidate"]) for row in scorer_rows if str(row["candidate"]) not in {"round6", *restart_default}],
    ]

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        scorer_row = _row_by_candidate(scorer_rows, "candidate", candidate)
        replay_row = _row_by_candidate(replay_rows, "case_candidate", candidate)
        rows.append(
            {
                "candidate": candidate,
                "scorer_verdict": str(scorer_row["verdict"]) if scorer_row is not None else None,
                "replay_verdict": str(replay_row["verdict"]) if replay_row is not None else None,
                "manifest_bucket": _manifest_bucket(
                    candidate,
                    str(scorer_row["verdict"]) if scorer_row is not None else None,
                    str(replay_row["verdict"]) if replay_row is not None else None,
                    restart_default,
                ),
            }
        )

    active = [row["candidate"] for row in rows if row["manifest_bucket"] == "active_benchmark"]
    default_restart = [row["candidate"] for row in rows if row["manifest_bucket"] == "default_restart_prior"]
    replay_alternates = [row["candidate"] for row in rows if row["manifest_bucket"] == "replay_validated_alternate"]
    hold_only = [row["candidate"] for row in rows if row["manifest_bucket"] == "hold_only_prior"]
    retired = [row["candidate"] for row in rows if row["manifest_bucket"] == "retired_prior"]
    seed_clean_unconfirmed = [row["candidate"] for row in rows if row["manifest_bucket"] == "seed_clean_unconfirmed_alternate"]

    lines = [
        "# Portfolio Frontier Manifest",
        "",
        f"- source campaign: `{campaign['name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        f"- active benchmark pack: `{seed_pack['benchmark']['active_candidate_pack']}`",
        "",
        "## Frontier Roles",
        "",
        "| Candidate | Seed-Pack Scorer | Frontier Replay | Manifest Bucket |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        scorer_verdict = row["scorer_verdict"] or "n/a"
        replay_verdict = row["replay_verdict"] or "n/a"
        lines.append(
            f"| `{row['candidate']}` | `{scorer_verdict}` | `{replay_verdict}` | `{row['manifest_bucket']}` |"
        )

    lines.extend(
        [
            "",
            "## Manifest Summary",
            "",
            f"- active benchmark: `{active}`",
            f"- default restart prior: `{default_restart}`",
            f"- replay-validated alternates: `{replay_alternates}`",
            f"- hold-only priors: `{hold_only}`",
            f"- retired priors: `{retired}`",
            f"- seed-clean but replay-unconfirmed alternates: `{seed_clean_unconfirmed}`",
            "",
            "## Interpretation",
            "",
            "- `round6` remains the active benchmark.",
            "- `round7` stays the default restart prior because it is the clean lowest-friction restart choice on the measured frontier.",
            "- `round10` is the only replay-validated alternate that survives the historically stitched support-plus-weakness surface, so it is the measured escalation target if the project wants a fuller alternate check.",
            "- `round5` stays hold-only: it is seed-clean on the packed frontier but still below the incumbent on broader dev mean.",
            "- `door3_post5` and `post_unlock_x5` remain retired priors.",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "rows": rows,
                "active": active,
                "default_restart": default_restart,
                "replay_validated_alternates": replay_alternates,
                "hold_only": hold_only,
                "retired": retired,
                "seed_clean_unconfirmed": seed_clean_unconfirmed,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a manifest over the current measured portfolio frontier")
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
