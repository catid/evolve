from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from psmn_rl.analysis.benchmark_pack import sha256_path
from psmn_rl.analysis.claim_gate import evaluate_pack_claim_gate_from_paths, render_pack_gate_report
from psmn_rl.analysis.lss_robustness import _format_float
from psmn_rl.utils.io import get_git_commit, get_git_dirty


LABELS: tuple[str, ...] = (
    "recovered_token_dense",
    "kl_lss_token_dense",
    "kl_lss_single_expert",
    "baseline_sare",
    "kl_lss_sare",
)
DISPLAY_NAMES: dict[str, str] = {
    "recovered_token_dense": "recovered token_dense",
    "kl_lss_token_dense": "KL learner-state token_dense",
    "kl_lss_single_expert": "KL learner-state single_expert",
    "baseline_sare": "baseline PPO SARE",
    "kl_lss_sare": "KL learner-state SARE",
}
STUDENT_TO_LABEL: dict[str, str] = {
    "sare": "kl_lss_sare",
    "token_dense": "kl_lss_token_dense",
    "single_expert": "kl_lss_single_expert",
}


@dataclass(slots=True)
class RunRecord:
    candidate: str
    lane: str
    seed: int
    label: str
    run_dir: Path
    summary: dict[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _float(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "-"):
        return default
    return float(value)


def _int(value: Any, default: int = 0) -> int:
    if value in (None, "", "-"):
        return default
    return int(value)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "complete_seed_failures": 0.0, "seed_count": 0.0}
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "complete_seed_failures": float(sum(1 for value in values if value <= 0.0)),
        "seed_count": float(len(values)),
    }


def _lane_seed_list(group: dict[str, Any]) -> list[tuple[str, int]]:
    if "lane" in group:
        return [(str(group["lane"]), int(seed)) for seed in group.get("seeds", [])]
    items: list[tuple[str, int]] = []
    for block in group.get("blocks", []):
        lane = str(block["lane"])
        items.extend((lane, int(seed)) for seed in block.get("seeds", []))
    return items


def _read_baseline_greedy_rows(path: Path) -> list[dict[str, Any]]:
    return [row for row in _read_csv_rows(path) if str(row.get("mode")) == "greedy"]


def _baseline_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    return {
        (str(row["lane"]), int(row["seed"]), str(row["label"])): row
        for row in rows
    }


def _summary_round(summary: dict[str, Any], round_index: int | None) -> dict[str, Any]:
    if round_index is None or round_index <= 0:
        return {}
    rounds = summary.get("rounds", [])
    if round_index > len(rounds):
        return {}
    return dict(rounds[round_index - 1])


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


def _records_by_candidate(records: list[RunRecord], label: str | None = None) -> dict[str, list[RunRecord]]:
    grouped: dict[str, list[RunRecord]] = {}
    for record in records:
        if label is not None and record.label != label:
            continue
        grouped.setdefault(record.candidate, []).append(record)
    return grouped


def _weak_baseline_by_seed(rows: list[dict[str, Any]], label: str) -> dict[int, float]:
    return {
        int(row["seed"]): _float(row["eval_success_rate"])
        for row in rows
        if str(row["lane"]) == "fresh_final" and str(row["label"]) == label
    }


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
        "eval_success_rate": _float(record.summary.get("final_greedy_success")),
        "final_greedy_success": _float(record.summary.get("final_greedy_success")),
        "best_round_index": best_round_index,
        "best_round_greedy_success": _float(record.summary.get("best_round_greedy_success")),
        "best_round_disagreement": _float(best_round.get("collection/disagreement_rate")),
        "best_round_unique_ratio": _float(best_round.get("collection/unique_state_ratio")),
        "best_round_post_unlock_frac": _float(best_round.get("collection/phase_frac_post_unlock")),
        "final_round_disagreement": _float(final_round.get("collection/disagreement_rate")),
        "final_round_unique_ratio": _float(final_round.get("collection/unique_state_ratio")),
        "final_round_post_unlock_frac": _float(final_round.get("collection/phase_frac_post_unlock")),
    }


def _candidate_means(records: list[RunRecord]) -> dict[str, float]:
    return {
        record.label: _float(record.summary.get("final_greedy_success"))
        for record in records
    }


def _artifact(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "sha256": sha256_path(path),
        "size_bytes": path.stat().st_size,
    }


def _render_registration(campaign: dict[str, Any], output: Path) -> None:
    lines = [
        "# Long Campaign Registration",
        "",
        f"- frozen manifest: `{campaign['frozen_manifest']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Intervention Family Shortlist",
        "",
    ]
    for candidate_id, candidate in campaign["candidates"].items():
        lines.append(f"- `{candidate_id}`: {candidate['hypothesis']}")
    lines.extend(
        [
            "",
            "## Seed Blocks",
            "",
            f"- weak block: `{campaign['seed_groups']['weak_block']['lane']}` seeds `{campaign['seed_groups']['weak_block']['seeds']}`",
            f"- original block: `{campaign['seed_groups']['original']['seeds']}`",
            f"- fresh block: `{campaign['seed_groups']['fresh']['seeds']}`",
            f"- fresh-extra block: `{campaign['seed_groups']['fresh_extra']['seeds']}`",
            "",
            "## Stage Gates",
            "",
            "- Stage 2: weak-block mean must beat `0.3125` and show a non-noisy per-seed improvement pattern.",
            "- Stage 3: candidate `SARE` must at least match same-block `single_expert` and not improve controls more strongly than routed `SARE`.",
            "- Stage 4: combined DoorKey `SARE` mean must stay at or above `0.7122` with no new complete-seed failures on previously healthy blocks.",
            "- Stage 5: fixed-router override and worst expert ablation must remain materially harmful on one improved weak seed and one strong seed.",
            "- Stage 6: candidate pack must clear the existing pack-based gate with no narrative override.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_baseline_sync(campaign: dict[str, Any], manifest: dict[str, Any], combined_csv: Path, final_block_csv: Path, output: Path) -> None:
    combined_rows = _read_baseline_greedy_rows(combined_csv)
    final_rows = _read_baseline_greedy_rows(final_block_csv)
    combined_lookup = _baseline_lookup(combined_rows)
    final_lookup = _baseline_lookup(final_rows)
    lines = [
        "# Long Campaign Baseline Sync",
        "",
        f"- frozen manifest: `{campaign['frozen_manifest']}`",
        f"- combined baseline csv: `{combined_csv}`",
        f"- weak-block baseline csv: `{final_block_csv}`",
        "",
        "## Weak Block",
        "",
        "| Seed | KL learner-state token_dense | KL learner-state single_expert | KL learner-state SARE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for seed in campaign["seed_groups"]["weak_block"]["seeds"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(seed),
                    _format_float(final_lookup[("fresh_final", int(seed), "kl_lss_token_dense")]["eval_success_rate"]),
                    _format_float(final_lookup[("fresh_final", int(seed), "kl_lss_single_expert")]["eval_success_rate"]),
                    _format_float(final_lookup[("fresh_final", int(seed), "kl_lss_sare")]["eval_success_rate"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Frozen Thresholds",
            "",
            f"- retry-block KL learner-state `SARE` mean: `{manifest['thresholds']['retry_block_means']['kl_lss_sare']:.4f}`",
            f"- retry-block KL learner-state `single_expert` mean: `{manifest['thresholds']['retry_block_means']['kl_lss_single_expert']:.4f}`",
            f"- combined KL learner-state `SARE` mean: `{manifest['thresholds']['combined_means']['kl_lss_sare']:.4f}`",
            "",
            "## Interpretation",
            "",
            "- The frozen retry block and combined DoorKey picture still match the current sealed claim envelope closely enough to start the staged campaign from the accepted baseline.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_mechanism_shortlist(campaign: dict[str, Any], output: Path) -> None:
    lines = [
        "# Long Campaign Mechanism Shortlist",
        "",
        "| Candidate | Intervention Family | Mechanism Hypothesis | Weak-Block Target | Expected Failure Mode |",
        "| --- | --- | --- | --- | --- |",
    ]
    for candidate_id, candidate in campaign["candidates"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{candidate_id}`",
                    str(candidate["intervention_family"]),
                    str(candidate["hypothesis"]),
                    ", ".join(str(seed) for seed in candidate["target_seeds"]),
                    str(candidate["expected_failure_mode"]),
                ]
            )
            + " |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stage2_pass(summary_rows: list[dict[str, Any]], baseline_by_seed: dict[int, float]) -> bool:
    values = [_float(row["final_greedy_success"]) for row in summary_rows]
    mean_value = _stats(values)["mean"]
    if mean_value <= 0.3125:
        return False
    zero_failures = all(value > 0.0 for value in values)
    seed_map = {int(row["seed"]): _float(row["final_greedy_success"]) for row in summary_rows}
    improved_47 = seed_map.get(47, 0.0) > baseline_by_seed.get(47, 0.0)
    preserved_53_59 = seed_map.get(53, 0.0) >= baseline_by_seed.get(53, 0.0) and seed_map.get(59, 0.0) >= baseline_by_seed.get(59, 0.0)
    improved_53_59 = ((seed_map.get(53, 0.0) + seed_map.get(59, 0.0)) / 2.0) > (
        (baseline_by_seed.get(53, 0.0) + baseline_by_seed.get(59, 0.0)) / 2.0
    )
    preserved_47 = seed_map.get(47, 0.0) >= baseline_by_seed.get(47, 0.0)
    return bool(zero_failures or (improved_47 and preserved_53_59) or (improved_53_59 and preserved_47))


def _stage3_pass(sare_stats: dict[str, float], single_stats: dict[str, float], token_delta: float, single_delta: float, sare_delta: float) -> bool:
    return sare_stats["mean"] >= single_stats["mean"] and sare_delta >= max(token_delta, single_delta)


def _render_stage2(records: list[RunRecord], baseline_rows: list[dict[str, Any]], output: Path, csv_output: Path, json_output: Path) -> None:
    baseline_by_seed = _weak_baseline_by_seed(baseline_rows, "kl_lss_sare")
    detail_rows = [_candidate_row(record) for record in records if record.label == "kl_lss_sare"]
    grouped = _records_by_candidate(records, label="kl_lss_sare")
    candidate_summaries: list[dict[str, Any]] = []
    for candidate, candidate_records in sorted(grouped.items()):
        seed_rows = [_candidate_row(record) for record in sorted(candidate_records, key=lambda item: item.seed)]
        stats = _stats([row["final_greedy_success"] for row in seed_rows])
        stage2_pass = _stage2_pass(seed_rows, baseline_by_seed)
        candidate_summaries.append(
            {
                "candidate": candidate,
                "mean_greedy_success": stats["mean"],
                "complete_seed_failures": stats["complete_seed_failures"],
                "packability_readiness": "ready" if stage2_pass else "weak_block_fail",
                "stage2_pass": stage2_pass,
            }
        )
    advancing_candidates = [
        row["candidate"]
        for row in sorted(
            [row for row in candidate_summaries if row["stage2_pass"]],
            key=lambda item: item["mean_greedy_success"],
            reverse=True,
        )[:2]
    ]
    lines = [
        "# Long Campaign Stage 2 Screening",
        "",
        "| Candidate | Seed | Final Greedy | Best Round | Best Round Greedy | Best Round Disagreement | Best Round Unique Ratio | Best Round Post-Unlock Frac | Packability |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(detail_rows, key=lambda item: (str(item["candidate"]), int(item["seed"]))):
        readiness = "ready" if row["candidate"] in advancing_candidates else "weak_block_fail"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['candidate']}`",
                    str(row["seed"]),
                    _format_float(row["final_greedy_success"]),
                    str(_int(row["best_round_index"])),
                    _format_float(row["best_round_greedy_success"]),
                    _format_float(row["best_round_disagreement"]),
                    _format_float(row["best_round_unique_ratio"]),
                    _format_float(row["best_round_post_unlock_frac"]),
                    readiness,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Candidate Summary",
            "",
            "| Candidate | Mean Greedy Success | Complete-Seed Failures | Stage 2 |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in sorted(candidate_summaries, key=lambda item: item["mean_greedy_success"], reverse=True):
        lines.append(
            f"| `{row['candidate']}` | `{row['mean_greedy_success']:.4f}` | `{int(row['complete_seed_failures'])}` | `{'pass' if row['stage2_pass'] else 'stop'}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- advancing candidates: `{advancing_candidates}`",
            "- Stage 2 keeps only candidates that beat the frozen weak-block `SARE` mean and show a non-noisy per-seed improvement pattern on `47/53/59`.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(csv_output, detail_rows)
    _write_json(
        json_output,
        {
            "stage": "stage2",
            "detail_rows": detail_rows,
            "candidate_summaries": candidate_summaries,
            "advancing_candidates": advancing_candidates,
        },
    )


def _render_stage3(
    records: list[RunRecord],
    manifest: dict[str, Any],
    stage2_payload: dict[str, Any],
    output: Path,
    csv_output: Path,
    json_output: Path,
) -> None:
    detail_rows = [_candidate_row(record) for record in records]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in detail_rows:
        grouped.setdefault((str(row["candidate"]), str(row["label"])), []).append(row)
    candidate_summaries: list[dict[str, Any]] = []
    for candidate in stage2_payload.get("advancing_candidates", []):
        sare_rows = grouped.get((candidate, "kl_lss_sare"), [])
        token_rows = grouped.get((candidate, "kl_lss_token_dense"), [])
        single_rows = grouped.get((candidate, "kl_lss_single_expert"), [])
        if not sare_rows or not token_rows or not single_rows:
            continue
        sare_stats = _stats([row["final_greedy_success"] for row in sare_rows])
        token_stats = _stats([row["final_greedy_success"] for row in token_rows])
        single_stats = _stats([row["final_greedy_success"] for row in single_rows])
        frozen_retry = manifest["thresholds"]["retry_block_means"]
        sare_delta = sare_stats["mean"] - float(frozen_retry["kl_lss_sare"])
        token_delta = token_stats["mean"] - float(frozen_retry["kl_lss_token_dense"])
        single_delta = single_stats["mean"] - float(frozen_retry["kl_lss_single_expert"])
        fairness_pass = _stage3_pass(sare_stats, single_stats, token_delta, single_delta, sare_delta)
        candidate_summaries.append(
            {
                "candidate": candidate,
                "sare_mean": sare_stats["mean"],
                "token_mean": token_stats["mean"],
                "single_mean": single_stats["mean"],
                "sare_delta": sare_delta,
                "token_delta": token_delta,
                "single_delta": single_delta,
                "stage3_pass": fairness_pass,
            }
        )
    survivors = [row["candidate"] for row in candidate_summaries if row["stage3_pass"]]
    best_candidate = None
    if survivors:
        best_candidate = max(
            [row for row in candidate_summaries if row["stage3_pass"]],
            key=lambda item: item["sare_mean"],
        )["candidate"]
    lines = [
        "# Long Campaign Stage 3 Fairness",
        "",
        "| Candidate | Variant | Seed | Final Greedy | Best Round | Best Round Greedy |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in sorted(detail_rows, key=lambda item: (str(item["candidate"]), str(item["label"]), int(item["seed"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['candidate']}`",
                    DISPLAY_NAMES[str(row["label"])],
                    str(row["seed"]),
                    _format_float(row["final_greedy_success"]),
                    str(_int(row["best_round_index"])),
                    _format_float(row["best_round_greedy_success"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Candidate Summary",
            "",
            "| Candidate | SARE Mean | token_dense Mean | single_expert Mean | SARE Delta | token Delta | single Delta | Stage 3 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in sorted(candidate_summaries, key=lambda item: item["sare_mean"], reverse=True):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['candidate']}`",
                    _format_float(row["sare_mean"]),
                    _format_float(row["token_mean"]),
                    _format_float(row["single_mean"]),
                    _format_float(row["sare_delta"]),
                    _format_float(row["token_delta"]),
                    _format_float(row["single_delta"]),
                    "`pass`" if row["stage3_pass"] else "`stop`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- stage-3 survivors: `{survivors}`",
            f"- best surviving candidate: `{best_candidate}`",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(csv_output, detail_rows)
    _write_json(
        json_output,
        {
            "stage": "stage3",
            "detail_rows": detail_rows,
            "candidate_summaries": candidate_summaries,
            "surviving_candidates": survivors,
            "best_candidate": best_candidate,
        },
    )


def _variant_seed_map(detail_rows: list[dict[str, Any]], label: str) -> dict[tuple[str, int], float]:
    return {(str(row["lane"]), int(row["seed"])): _float(row["final_greedy_success"]) for row in detail_rows if str(row["label"]) == label}


def _render_stage4(
    manifest: dict[str, Any],
    baseline_combined_rows: list[dict[str, Any]],
    weak_rows: list[dict[str, Any]],
    strong_records: list[RunRecord],
    best_candidate: str,
    output: Path,
    csv_output: Path,
    json_output: Path,
) -> None:
    strong_rows = [_candidate_row(record) for record in strong_records]
    detail_rows = [row for row in weak_rows if row["candidate"] == best_candidate] + strong_rows
    candidate_variant_stats: dict[str, dict[str, float]] = {}
    for label in ("kl_lss_sare", "kl_lss_token_dense", "kl_lss_single_expert"):
        candidate_variant_stats[label] = _stats([_float(row["final_greedy_success"]) for row in detail_rows if str(row["label"]) == label])
    baseline_lookup = _baseline_lookup(baseline_combined_rows)
    healthy_strong_failures: list[tuple[str, int]] = []
    sare_map = _variant_seed_map(detail_rows, "kl_lss_sare")
    for lane, seeds in (
        ("original", [7, 11, 19]),
        ("fresh", [23, 29, 31]),
        ("fresh_extra", [37, 41, 43]),
    ):
        for seed in seeds:
            frozen_value = _float(baseline_lookup[(lane, seed, "kl_lss_sare")]["eval_success_rate"])
            if frozen_value > 0.0 and sare_map.get((lane, seed), 0.0) <= 0.0:
                healthy_strong_failures.append((lane, seed))
    combined_pass = (
        candidate_variant_stats["kl_lss_sare"]["mean"] >= float(manifest["thresholds"]["combined_means"]["kl_lss_sare"])
        and candidate_variant_stats["kl_lss_sare"]["complete_seed_failures"] <= float(manifest["thresholds"]["combined_complete_seed_failures"]["kl_lss_sare"])
        and not healthy_strong_failures
    )
    strong_sare_rows = [row for row in detail_rows if str(row["label"]) == "kl_lss_sare" and str(row["lane"]) != "fresh_final"]
    selected_strong = max(strong_sare_rows, key=lambda item: item["final_greedy_success"]) if strong_sare_rows else {"lane": "original", "seed": 7}
    weak_sare_rows = [row for row in detail_rows if str(row["label"]) == "kl_lss_sare" and str(row["lane"]) == "fresh_final"]
    selected_weak = max(weak_sare_rows, key=lambda item: item["final_greedy_success"]) if weak_sare_rows else {"lane": "fresh_final", "seed": 47}
    lines = [
        "# Long Campaign Stage 4 Replication",
        "",
        f"- candidate: `{best_candidate}`",
        "",
        "| Lane | Seed | Variant | Final Greedy | Best Round | Best Round Greedy |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in sorted(detail_rows, key=lambda item: (str(item["lane"]), int(item["seed"]), str(item["label"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["lane"]),
                    str(row["seed"]),
                    DISPLAY_NAMES[str(row["label"])],
                    _format_float(row["final_greedy_success"]),
                    str(_int(row["best_round_index"])),
                    _format_float(row["best_round_greedy_success"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Candidate Combined Summary",
            "",
            "| Variant | Mean Greedy Success | Complete-Seed Failures |",
            "| --- | ---: | ---: |",
            f"| KL learner-state token_dense | `{candidate_variant_stats['kl_lss_token_dense']['mean']:.4f}` | `{int(candidate_variant_stats['kl_lss_token_dense']['complete_seed_failures'])}` |",
            f"| KL learner-state single_expert | `{candidate_variant_stats['kl_lss_single_expert']['mean']:.4f}` | `{int(candidate_variant_stats['kl_lss_single_expert']['complete_seed_failures'])}` |",
            f"| KL learner-state SARE | `{candidate_variant_stats['kl_lss_sare']['mean']:.4f}` | `{int(candidate_variant_stats['kl_lss_sare']['complete_seed_failures'])}` |",
            "",
            "## Interpretation",
            "",
            f"- selected weak case for route validation: `({selected_weak['lane']}, {selected_weak['seed']})`",
            f"- selected strong case for route validation: `({selected_strong['lane']}, {selected_strong['seed']})`",
            f"- new complete-seed failures on previously healthy strong seeds: `{healthy_strong_failures}`",
            f"- stage-4 status: `{'pass' if combined_pass else 'stop'}`",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(csv_output, detail_rows)
    _write_json(
        json_output,
        {
            "stage": "stage4",
            "candidate": best_candidate,
            "detail_rows": detail_rows,
            "candidate_variant_stats": candidate_variant_stats,
            "healthy_strong_failures": healthy_strong_failures,
            "selected_weak_case": {"lane": selected_weak["lane"], "seed": selected_weak["seed"]},
            "selected_strong_case": {"lane": selected_strong["lane"], "seed": selected_strong["seed"]},
            "stage4_pass": combined_pass,
        },
    )


def _render_stage4_skipped(output: Path, csv_output: Path, json_output: Path, reason: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            [
                "# Long Campaign Stage 4 Replication",
                "",
                "## Status",
                "",
                f"- skipped: `{reason}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_output, [])
    _write_json(
        json_output,
        {
            "stage": "stage4",
            "candidate": None,
            "detail_rows": [],
            "candidate_variant_stats": {},
            "healthy_strong_failures": [],
            "selected_weak_case": None,
            "selected_strong_case": None,
            "stage4_pass": False,
            "skipped": True,
            "reason": reason,
        },
    )


def _render_stage5(route_rows: list[dict[str, Any]], candidate: str, output: Path, json_output: Path) -> None:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in route_rows:
        grouped.setdefault((str(row["lane"]), int(row["seed"])), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (lane, seed), case_rows in sorted(grouped.items()):
        baseline = next(row for row in case_rows if row["probe"] == "baseline")
        fixed = next(row for row in case_rows if row["probe"] == "router_override")
        random = next(row for row in case_rows if row["probe"] == "route_randomization")
        ablations = [row for row in case_rows if row["probe"] == "expert_ablation"]
        worst = min(ablations, key=lambda item: _float(item["eval_success_rate"])) if ablations else baseline
        summaries.append(
            {
                "lane": lane,
                "seed": seed,
                "baseline": _float(baseline["eval_success_rate"]),
                "fixed_drop": _float(baseline["eval_success_rate"]) - _float(fixed["eval_success_rate"]),
                "random_drop": _float(baseline["eval_success_rate"]) - _float(random["eval_success_rate"]),
                "worst_ablation_drop": _float(baseline["eval_success_rate"]) - _float(worst["eval_success_rate"]),
            }
        )
    fixed_ok = all(row["fixed_drop"] >= 0.25 for row in summaries)
    ablation_ok = all(row["worst_ablation_drop"] >= 0.25 for row in summaries)
    stage5_pass = bool(summaries and fixed_ok and ablation_ok)
    lines = [
        "# Long Campaign Stage 5 Route Validation",
        "",
        f"- candidate: `{candidate}`",
        "",
        "| Lane | Seed | Baseline | Fixed-Router Drop | Route-Randomization Drop | Worst Ablation Drop |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["lane"]),
                    str(row["seed"]),
                    _format_float(row["baseline"]),
                    _format_float(row["fixed_drop"]),
                    _format_float(row["random_drop"]),
                    _format_float(row["worst_ablation_drop"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", "", f"- stage-5 status: `{'pass' if stage5_pass else 'stop'}`"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(
        json_output,
        {
            "stage": "stage5",
            "candidate": candidate,
            "summaries": summaries,
            "stage5_pass": stage5_pass,
        },
    )


def _metrics_block(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for label in LABELS:
        values = [_float(row["eval_success_rate"]) for row in rows if str(row["label"]) == label]
        result[label] = _stats(values)
    return result


def _pack_metric_rows(rows: list[dict[str, Any]], lane_seeds: list[tuple[str, int]]) -> list[dict[str, Any]]:
    lane_seed_set = {(lane, seed) for lane, seed in lane_seeds}
    return [row for row in rows if (str(row["lane"]), int(row["seed"])) in lane_seed_set]


def _dedupe_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["lane"]), int(row["seed"]), str(row["label"]))
        deduped.setdefault(key, row)
    return list(deduped.values())


def _render_candidate_reports(
    candidate: str,
    weak_rows: list[dict[str, Any]],
    strong_rows: list[dict[str, Any]],
    baseline_combined_rows: list[dict[str, Any]],
    combined_md: Path,
    combined_csv: Path,
    retry_md: Path,
    retry_csv: Path,
    summary_md: Path,
    metrics_json: Path,
    candidate_pack_path: Path,
    frozen_pack_path: Path,
    manifest: dict[str, Any],
) -> None:
    candidate_rows = _dedupe_metric_rows(weak_rows + strong_rows)
    baseline_rows = [row for row in baseline_combined_rows if str(row["label"]) in {"recovered_token_dense", "baseline_sare"}]
    combined_lane_seeds = _lane_seed_list(manifest["seed_groups"]["combined"])
    retry_lane_seeds = _lane_seed_list(manifest["seed_groups"]["retry_block"])
    combined_rows = _dedupe_metric_rows(_pack_metric_rows(candidate_rows + baseline_rows, combined_lane_seeds))
    retry_rows = _dedupe_metric_rows(_pack_metric_rows(candidate_rows + baseline_rows, retry_lane_seeds))
    combined_metrics = _metrics_block(combined_rows)
    retry_metrics = _metrics_block(retry_rows)

    combined_lines = [
        "# Long Campaign Combined DoorKey Report",
        "",
        f"- candidate: `{candidate}`",
        "",
        "| Lane | Seed | Variant | Greedy Success |",
        "| --- | --- | --- | ---: |",
    ]
    for row in sorted(combined_rows, key=lambda item: (str(item["lane"]), int(item["seed"]), str(item["label"]))):
        combined_lines.append(
            f"| {row['lane']} | {row['seed']} | {DISPLAY_NAMES[str(row['label'])]} | {_format_float(_float(row['eval_success_rate']))} |"
        )
    combined_lines.extend(["", "## Summary", "", "| Variant | Mean | Complete-Seed Failures |", "| --- | ---: | ---: |"])
    for label in LABELS:
        combined_lines.append(
            f"| {DISPLAY_NAMES[label]} | `{combined_metrics[label]['mean']:.4f}` | `{int(combined_metrics[label]['complete_seed_failures'])}` |"
        )
    combined_md.parent.mkdir(parents=True, exist_ok=True)
    combined_md.write_text("\n".join(combined_lines) + "\n", encoding="utf-8")
    _write_csv(combined_csv, combined_rows)

    retry_lines = [
        "# Long Campaign Retry-Block Report",
        "",
        f"- candidate: `{candidate}`",
        "",
        "| Seed | recovered token_dense | KL learner-state token_dense | KL learner-state single_expert | baseline PPO SARE | KL learner-state SARE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lookup = _baseline_lookup(retry_rows)
    for _lane, seed in retry_lane_seeds:
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
    retry_lines.extend(["", "## Summary", "", "| Variant | Mean | Complete-Seed Failures |", "| --- | ---: | ---: |"])
    for label in LABELS:
        retry_lines.append(
            f"| {DISPLAY_NAMES[label]} | `{retry_metrics[label]['mean']:.4f}` | `{int(retry_metrics[label]['complete_seed_failures'])}` |"
        )
    retry_md.parent.mkdir(parents=True, exist_ok=True)
    retry_md.write_text("\n".join(retry_lines) + "\n", encoding="utf-8")
    _write_csv(retry_csv, retry_rows)

    summary_lines = [
        "# Long Campaign Candidate Summary",
        "",
        f"- candidate: `{candidate}`",
        f"- retry-block KL learner-state `SARE` mean: `{retry_metrics['kl_lss_sare']['mean']:.4f}`",
        f"- retry-block KL learner-state `single_expert` mean: `{retry_metrics['kl_lss_single_expert']['mean']:.4f}`",
        f"- combined KL learner-state `SARE` mean: `{combined_metrics['kl_lss_sare']['mean']:.4f}`",
    ]
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    metrics_payload = {
        "schema_version": 1,
        "candidate_name": candidate,
        "task": "DoorKey",
        "evaluation": {"path_key": "external_policy_diagnostics", "episodes": 64},
        "requested_claims": ["bounded_teacher_guided_doorkey_sare"],
        "controls_present": list(LABELS),
        "metrics": {"combined": combined_metrics, "retry_block": retry_metrics},
        "actual_sets": {
            "combined_lane_seeds": [[lane, seed] for lane, seed in combined_lane_seeds],
            "retry_block_lane_seeds": [[lane, seed] for lane, seed in retry_lane_seeds],
        },
        "provenance": {"git_commit": get_git_commit(), "git_dirty": get_git_dirty(), "notes": f"long campaign candidate `{candidate}`"},
    }
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

    frozen_pack = _read_json(frozen_pack_path)
    candidate_pack = {
        "schema_version": 1,
        "pack_type": "candidate_result_pack",
        "candidate_name": candidate,
        "frozen_pack_reference": {
            "path": str(frozen_pack_path),
            "sha256": sha256_path(frozen_pack_path),
            "claim_id": frozen_pack["claim"]["id"],
        },
        "task": "DoorKey",
        "evaluation": {"path_key": "external_policy_diagnostics", "episodes": 64},
        "requested_claims": ["bounded_teacher_guided_doorkey_sare"],
        "controls_present": list(LABELS),
        "metrics": {"combined": combined_metrics, "retry_block": retry_metrics},
        "actual_sets": {
            "combined_lane_seeds": [[lane, seed] for lane, seed in combined_lane_seeds],
            "retry_block_lane_seeds": [[lane, seed] for lane, seed in retry_lane_seeds],
        },
        "artifacts": [
            _artifact(summary_md, "candidate_summary_markdown"),
            _artifact(metrics_json, "candidate_metrics_json"),
            _artifact(combined_md, "combined_report_markdown"),
            _artifact(combined_csv, "combined_report_csv"),
            _artifact(retry_md, "retry_block_report_markdown"),
            _artifact(retry_csv, "retry_block_report_csv"),
        ],
        "provenance": {"git_commit": get_git_commit(), "git_dirty": get_git_dirty(), "notes": f"long campaign candidate `{candidate}`"},
    }
    candidate_pack_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_pack_path.write_text(json.dumps(candidate_pack, indent=2, sort_keys=True), encoding="utf-8")


def _render_decision_memo(
    stage2_payload: dict[str, Any],
    stage3_payload: dict[str, Any] | None,
    stage4_payload: dict[str, Any] | None,
    stage5_payload: dict[str, Any] | None,
    gate_payload: dict[str, Any] | None,
    output: Path,
) -> None:
    stop_stage = None
    final_status = "FAIL: claim remains frozen"
    if not stage2_payload.get("advancing_candidates"):
        stop_stage = "Stage 2"
    elif stage3_payload is None or not stage3_payload.get("best_candidate"):
        stop_stage = "Stage 3"
    elif stage4_payload is None or not stage4_payload.get("stage4_pass"):
        stop_stage = "Stage 4"
    elif stage5_payload is None or not stage5_payload.get("stage5_pass"):
        stop_stage = "Stage 5"
    elif gate_payload is None:
        stop_stage = "Stage 6"
    else:
        final_status = str(gate_payload.get("verdict"))
        if final_status != "PASS: thaw consideration allowed":
            stop_stage = "Stage 6"

    lines = [
        "# Long Campaign Decision Memo",
        "",
        f"- stop stage: `{stop_stage or 'none'}`",
        f"- final gate status: `{final_status}`",
        "",
        "## Stage Summary",
        "",
        f"- stage-2 advancing candidates: `{stage2_payload.get('advancing_candidates', [])}`",
        f"- stage-3 best candidate: `{None if stage3_payload is None else stage3_payload.get('best_candidate')}`",
        f"- stage-4 pass: `{None if stage4_payload is None else stage4_payload.get('stage4_pass')}`",
        f"- stage-5 pass: `{None if stage5_payload is None else stage5_payload.get('stage5_pass')}`",
        "",
        "## Final Result",
        "",
    ]
    if final_status == "PASS: thaw consideration allowed":
        lines.append("- The campaign produced a valid candidate pack that clears the existing frozen DoorKey gate. Thaw consideration is now allowed within DoorKey only.")
    else:
        lines.append("- The campaign did not produce a candidate that clears the existing frozen DoorKey gate. The claim remains frozen, now with campaign-backed negative evidence.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Long-horizon DoorKey thaw campaign reporting.")
    sub = parser.add_subparsers(dest="command", required=True)

    registration = sub.add_parser("registration")
    registration.add_argument("--campaign-config", required=True)
    registration.add_argument("--output", required=True)

    baseline = sub.add_parser("baseline-sync")
    baseline.add_argument("--campaign-config", required=True)
    baseline.add_argument("--combined-csv", required=True)
    baseline.add_argument("--final-csv", required=True)
    baseline.add_argument("--output", required=True)

    shortlist = sub.add_parser("mechanism-shortlist")
    shortlist.add_argument("--campaign-config", required=True)
    shortlist.add_argument("--output", required=True)

    stage2 = sub.add_parser("stage2-report")
    stage2.add_argument("--stage2-root", required=True)
    stage2.add_argument("--baseline-final-csv", required=True)
    stage2.add_argument("--output", required=True)
    stage2.add_argument("--csv", required=True)
    stage2.add_argument("--json", required=True)

    stage3 = sub.add_parser("stage3-report")
    stage3.add_argument("--manifest", required=True)
    stage3.add_argument("--stage3-root", required=True)
    stage3.add_argument("--stage2-json", required=True)
    stage3.add_argument("--output", required=True)
    stage3.add_argument("--csv", required=True)
    stage3.add_argument("--json", required=True)

    stage4 = sub.add_parser("stage4-report")
    stage4.add_argument("--manifest", required=True)
    stage4.add_argument("--baseline-combined-csv", required=True)
    stage4.add_argument("--stage3-csv", required=True)
    stage4.add_argument("--stage3-json", required=True)
    stage4.add_argument("--stage4-root", required=True)
    stage4.add_argument("--output", required=True)
    stage4.add_argument("--csv", required=True)
    stage4.add_argument("--json", required=True)

    stage5 = sub.add_parser("stage5-report")
    stage5.add_argument("--candidate", required=True)
    stage5.add_argument("--route-csv", required=True)
    stage5.add_argument("--output", required=True)
    stage5.add_argument("--json", required=True)

    pack = sub.add_parser("candidate-pack")
    pack.add_argument("--manifest", required=True)
    pack.add_argument("--frozen-pack", required=True)
    pack.add_argument("--baseline-combined-csv", required=True)
    pack.add_argument("--stage3-csv", required=True)
    pack.add_argument("--stage3-json", required=True)
    pack.add_argument("--stage4-csv", required=True)
    pack.add_argument("--candidate-summary-output", required=True)
    pack.add_argument("--candidate-metrics-output", required=True)
    pack.add_argument("--combined-report-output", required=True)
    pack.add_argument("--combined-report-csv", required=True)
    pack.add_argument("--retry-report-output", required=True)
    pack.add_argument("--retry-report-csv", required=True)
    pack.add_argument("--candidate-pack-output", required=True)

    memo = sub.add_parser("decision-memo")
    memo.add_argument("--stage2-json", required=True)
    memo.add_argument("--stage3-json")
    memo.add_argument("--stage4-json")
    memo.add_argument("--stage5-json")
    memo.add_argument("--gate-json")
    memo.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "registration":
        _render_registration(_load_yaml(Path(args.campaign_config)), Path(args.output))
        return
    if args.command == "baseline-sync":
        campaign = _load_yaml(Path(args.campaign_config))
        manifest = _load_yaml(Path(campaign["frozen_manifest"]))
        _render_baseline_sync(campaign, manifest, Path(args.combined_csv), Path(args.final_csv), Path(args.output))
        return
    if args.command == "mechanism-shortlist":
        _render_mechanism_shortlist(_load_yaml(Path(args.campaign_config)), Path(args.output))
        return
    if args.command == "stage2-report":
        records = _discover_runs(Path(args.stage2_root))
        baseline_rows = _read_baseline_greedy_rows(Path(args.baseline_final_csv))
        _render_stage2(records, baseline_rows, Path(args.output), Path(args.csv), Path(args.json))
        return
    if args.command == "stage3-report":
        manifest = _load_yaml(Path(args.manifest))
        stage2_payload = _read_json(Path(args.stage2_json))
        records = _discover_runs(Path(args.stage3_root))
        _render_stage3(records, manifest, stage2_payload, Path(args.output), Path(args.csv), Path(args.json))
        return
    if args.command == "stage4-report":
        manifest = _load_yaml(Path(args.manifest))
        baseline_rows = _read_baseline_greedy_rows(Path(args.baseline_combined_csv))
        weak_rows = _read_csv_rows(Path(args.stage3_csv))
        stage3_payload = _read_json(Path(args.stage3_json))
        best_candidate = stage3_payload.get("best_candidate")
        if not best_candidate:
            _render_stage4_skipped(Path(args.output), Path(args.csv), Path(args.json), "no candidate survived stage 3 fairness")
            return
        strong_records = [record for record in _discover_runs(Path(args.stage4_root)) if record.candidate == best_candidate]
        _render_stage4(manifest, baseline_rows, weak_rows, strong_records, str(best_candidate), Path(args.output), Path(args.csv), Path(args.json))
        return
    if args.command == "stage5-report":
        route_rows = _read_csv_rows(Path(args.route_csv))
        _render_stage5(route_rows, args.candidate, Path(args.output), Path(args.json))
        return
    if args.command == "candidate-pack":
        manifest = _load_yaml(Path(args.manifest))
        baseline_rows = _read_baseline_greedy_rows(Path(args.baseline_combined_csv))
        stage3_rows = [row for row in _read_csv_rows(Path(args.stage3_csv)) if str(row["label"]) in {"kl_lss_sare", "kl_lss_token_dense", "kl_lss_single_expert"}]
        stage4_rows = [row for row in _read_csv_rows(Path(args.stage4_csv)) if str(row["label"]) in {"kl_lss_sare", "kl_lss_token_dense", "kl_lss_single_expert"}]
        stage3_payload = _read_json(Path(args.stage3_json))
        candidate = stage3_payload.get("best_candidate")
        if not candidate:
            raise SystemExit("stage3 did not produce a best candidate")
        weak_rows = [row for row in stage3_rows if str(row["candidate"]) == str(candidate)]
        strong_rows = [row for row in stage4_rows if str(row["candidate"]) == str(candidate)]
        _render_candidate_reports(
            str(candidate),
            weak_rows,
            strong_rows,
            baseline_rows,
            Path(args.combined_report_output),
            Path(args.combined_report_csv),
            Path(args.retry_report_output),
            Path(args.retry_report_csv),
            Path(args.candidate_summary_output),
            Path(args.candidate_metrics_output),
            Path(args.candidate_pack_output),
            Path(args.frozen_pack),
            manifest,
        )
        return
    if args.command == "decision-memo":
        stage2_payload = _read_json(Path(args.stage2_json))
        stage3_payload = _read_json(Path(args.stage3_json)) if args.stage3_json else None
        stage4_payload = _read_json(Path(args.stage4_json)) if args.stage4_json else None
        stage5_payload = _read_json(Path(args.stage5_json)) if args.stage5_json else None
        gate_payload = _read_json(Path(args.gate_json)) if args.gate_json else None
        _render_decision_memo(stage2_payload, stage3_payload, stage4_payload, stage5_payload, gate_payload, Path(args.output))
        return


if __name__ == "__main__":
    main()
