from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from psmn_rl.analysis.policy_diagnostics import DEFAULT_MODES
from psmn_rl.config import load_config
from psmn_rl.logging import configure_logging
from psmn_rl.models.factory import build_model
from psmn_rl.rl.distributed.ddp import cleanup_distributed, init_distributed
from psmn_rl.rl.ppo.algorithm import collect_policy_diagnostics
from psmn_rl.utils.io import get_git_commit, get_git_dirty
from psmn_rl.utils.seed import set_seed


SEED_PATTERN = re.compile(r"seed_(\d+)")


@dataclass(slots=True)
class EvalTarget:
    seed: int
    label: str
    variant: str
    config_path: Path
    checkpoint_path: Path
    run_dir: Path
    method: str
    stage: str
    round_index: int | None = None
    round_diagnostics: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    command_path: Path | None = None


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_read_command(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def _best_sampled(rows: list[dict[str, Any]]) -> tuple[str, float]:
    sampled = [row for row in rows if row["mode"] != "greedy"]
    if not sampled:
        greedy = next(row for row in rows if row["mode"] == "greedy")
        return "greedy", float(greedy.get("eval_success_rate", 0.0))
    best = max(sampled, key=lambda row: float(row.get("eval_success_rate", 0.0)))
    return str(best["mode"]), float(best.get("eval_success_rate", 0.0))


def _greedy_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next(row for row in rows if row["mode"] == "greedy")


def _seed_from_path(path: Path) -> int:
    for part in path.parts:
        match = SEED_PATTERN.fullmatch(part)
        if match:
            return int(match.group(1))
    raise ValueError(f"could not infer seed from path: {path}")


def _evaluate_targets(targets: list[EvalTarget], device: str, episodes: int) -> list[dict[str, Any]]:
    if not targets:
        return []
    ctx = init_distributed(device, "auto")
    configure_logging(ctx.is_main_process)
    rows: list[dict[str, Any]] = []
    try:
        for target in targets:
            config = load_config(target.config_path)
            config.system.device = device
            config.logging.tensorboard = False
            set_seed(config.seed + ctx.rank, deterministic=config.system.deterministic)
            envs = None
            try:
                from psmn_rl.envs.registry import make_vector_env

                envs = make_vector_env(config.env, seed=config.seed, world_rank=ctx.rank)
                model = build_model(config.model, envs.single_observation_space, envs.single_action_space).to(ctx.device)
            finally:
                if envs is not None:
                    envs.close()
            checkpoint = torch.load(target.checkpoint_path, map_location=ctx.device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            for mode_name, greedy, temperature in DEFAULT_MODES:
                diagnostics = collect_policy_diagnostics(
                    config=config,
                    model=model,
                    ctx=ctx,
                    episodes=episodes,
                    greedy=greedy,
                    temperature=temperature,
                    trace_limit=0,
                )
                row = {
                    "seed": target.seed,
                    "label": target.label,
                    "variant": target.variant,
                    "method": target.method,
                    "stage": target.stage,
                    "round_index": target.round_index,
                    "config_path": str(target.config_path),
                    "checkpoint_path": str(target.checkpoint_path),
                    "run_dir": str(target.run_dir),
                    "mode": mode_name,
                    "greedy": greedy,
                    "temperature": temperature,
                    **diagnostics.metrics,
                }
                if target.round_diagnostics:
                    row.update(target.round_diagnostics)
                if target.metadata:
                    row.update(target.metadata)
                rows.append(row)
        return rows if ctx.is_main_process else []
    finally:
        cleanup_distributed(ctx)


def _write_main_process_report(rows: list[dict[str, Any]], output: str, csv_output: str | None, content: str) -> None:
    if not rows:
        return
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    _write_csv(rows, csv_output)


def _render_run_config(
    base_config_path: Path,
    seed: int,
    run_name: str,
    output_dir: Path,
    target_path: Path,
) -> None:
    import yaml

    raw = yaml.safe_load(base_config_path.read_text(encoding="utf-8")) or {}
    raw["seed"] = seed
    raw.setdefault("logging", {})
    raw["logging"]["run_name"] = run_name
    raw["logging"]["output_dir"] = str(output_dir)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _write_csv(rows: list[dict[str, Any]], path: str | None) -> None:
    if path is None:
        return
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _baseline_targets(root: Path) -> list[EvalTarget]:
    variants = [
        ("flat_dense", "flat_dense_ent1e3"),
        ("recovered_token_dense", "token_dense_ent1e3"),
        ("baseline_sare", "sare_ent1e3"),
        ("learner_state_sare", "flat_dense_to_sare_lss"),
    ]
    targets: list[EvalTarget] = []
    for seed_dir in sorted(root.glob("seed_*")):
        seed = _seed_from_path(seed_dir)
        for label, run_name in variants:
            run_dir = seed_dir / run_name
            if not (run_dir / "latest.pt").exists():
                continue
            summary_path = run_dir / "summary.json"
            variant = run_name
            if summary_path.exists():
                summary = _read_json(summary_path)
                variant = str(summary.get("student_variant") or summary.get("teacher_variant") or run_name)
            else:
                config = load_config(run_dir / "resolved_config.yaml")
                variant = config.model.variant
            targets.append(
                EvalTarget(
                    seed=seed,
                    label=label,
                    variant=variant,
                    config_path=run_dir / "resolved_config.yaml",
                    checkpoint_path=run_dir / "latest.pt",
                    run_dir=run_dir,
                    method="baseline",
                    stage="baseline",
                    command_path=run_dir / "command.txt",
                )
            )
    return targets


def _lss_round_targets(root: Path) -> list[EvalTarget]:
    targets: list[EvalTarget] = []
    for run_dir in sorted(root.glob("seed_*/flat_dense_to_sare_lss")):
        seed = _seed_from_path(run_dir)
        summary = _read_json(run_dir / "summary.json")
        config_path = run_dir / "student_resolved_config.yaml"
        targets.append(
            EvalTarget(
                seed=seed,
                label=f"seed_{seed}_round_0",
                variant="sare",
                config_path=config_path,
                checkpoint_path=Path(summary["student_checkpoint"]),
                run_dir=run_dir,
                method="lss_baseline",
                stage="heterogeneity",
                round_index=0,
                command_path=run_dir / "command.txt",
            )
        )
        for round_index in range(1, len(summary["rounds"]) + 1):
            dataset_path = run_dir / f"round_{round_index:02d}_dataset.pt"
            diagnostics = {}
            if dataset_path.exists():
                dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
                diagnostics = dict(dataset.get("round_diagnostics") or {})
            targets.append(
                EvalTarget(
                    seed=seed,
                    label=f"seed_{seed}_round_{round_index}",
                    variant="sare",
                    config_path=config_path,
                    checkpoint_path=run_dir / f"round_{round_index:02d}.pt",
                    run_dir=run_dir,
                    method="lss_baseline",
                    stage="heterogeneity",
                    round_index=round_index,
                    round_diagnostics=diagnostics,
                    command_path=run_dir / "command.txt",
                )
            )
    return targets


def _sweep_targets(paths: list[str]) -> list[EvalTarget]:
    targets: list[EvalTarget] = []
    for raw in paths:
        root = Path(raw)
        for run_dir in sorted(root.glob("seed_*/*")):
            if not (run_dir / "latest.pt").exists() or not (run_dir / "summary.json").exists():
                continue
            seed = _seed_from_path(run_dir)
            summary = _read_json(run_dir / "summary.json")
            config_path = run_dir / "student_resolved_config.yaml"
            if not config_path.exists():
                config_path = run_dir / "resolved_config.yaml"
            method = run_dir.name
            targets.append(
                EvalTarget(
                    seed=seed,
                    label=method,
                    variant=str(summary.get("student_variant", "sare")),
                    config_path=config_path,
                    checkpoint_path=run_dir / "latest.pt",
                    run_dir=run_dir,
                    method=method,
                    stage=root.name,
                    metadata={
                        "loss": summary.get("loss"),
                        "aggregation": summary.get("aggregation"),
                        "weighting": summary.get("weighting"),
                        "target": summary.get("target"),
                        "max_dataset_steps": summary.get("max_dataset_steps"),
                    },
                    command_path=run_dir / "command.txt",
                )
            )
    return targets


def _build_reproduction_note(rows: list[dict[str, Any]], root: Path, episodes: int) -> str:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["seed"]), str(row["label"])), []).append(row)
    labels = [
        ("flat_dense", "flat_dense"),
        ("recovered_token_dense", "recovered token_dense"),
        ("baseline_sare", "baseline SARE"),
        ("learner_state_sare", "learner-state SARE"),
    ]
    lines = [
        "# Learner-State Robustness Reproduction Note",
        "",
        f"- output root: `{root}`",
        f"- external evaluation episodes per mode: `{episodes}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Per-Seed Results",
        "",
        "| Seed | Variant | Greedy Success | Best Sampled Success | Best Sampled Mode | Config | Checkpoint | Command |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for seed in sorted({key[0] for key in grouped}):
        for label, display in labels:
            mode_rows = grouped[(seed, label)]
            greedy = _greedy_row(mode_rows)
            best_mode, best_success = _best_sampled(mode_rows)
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(seed),
                        display,
                        _format_float(greedy.get("eval_success_rate")),
                        _format_float(best_success),
                        best_mode,
                        f"`{greedy['config_path']}`",
                        f"`{greedy['checkpoint_path']}`",
                        f"`{_maybe_read_command(Path(greedy['run_dir']) / 'command.txt') or '-'}`",
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This note reproduces the current multi-seed DoorKey teacher/student baseline on the external 64-episode decision path.",
            "- The routed comparison is only meaningful if these numbers stay close to the published teacher-extraction lane.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_heterogeneity_report(rows: list[dict[str, Any]], episodes: int) -> str:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped[(int(row["seed"]), int(row["round_index"] or 0))] = [*grouped.get((int(row["seed"]), int(row["round_index"] or 0)), []), row]
    lines = [
        "# Learner-State Seed Heterogeneity Report",
        "",
        f"- external evaluation episodes per mode: `{episodes}`",
        "",
        "| Seed | Round | Greedy Success | Best Sampled Success | Added Steps | Unique Ratio | Teacher Conf | Disagreement | Collection Route Entropy | Eval Route Entropy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    best_by_seed: dict[int, dict[str, Any]] = {}
    for (seed, round_index), mode_rows in sorted(grouped.items()):
        greedy = _greedy_row(mode_rows)
        best_mode, best_sampled = _best_sampled(mode_rows)
        candidate = (
            float(greedy.get("eval_success_rate", 0.0)),
            float(best_sampled),
            round_index,
        )
        incumbent = (
            float(best_by_seed[seed].get("greedy_success", -1.0)),
            float(best_by_seed[seed].get("best_sampled_success", -1.0)),
            int(best_by_seed[seed].get("round", -1)),
        ) if seed in best_by_seed else None
        if incumbent is None or candidate > incumbent:
            best_by_seed[seed] = {
                "round": round_index,
                "greedy_success": float(greedy.get("eval_success_rate", 0.0)),
                "best_sampled_success": best_sampled,
                "unique_ratio": greedy.get("collection/unique_state_ratio"),
                "teacher_confidence": greedy.get("collection/teacher_confidence_mean"),
                "disagreement": greedy.get("collection/disagreement_rate"),
            }
        lines.append(
            "| "
            + " | ".join(
                [
                    str(seed),
                    str(round_index),
                    _format_float(greedy.get("eval_success_rate")),
                    _format_float(best_sampled),
                    _format_float(greedy.get("collection/steps")),
                    _format_float(greedy.get("collection/unique_state_ratio")),
                    _format_float(greedy.get("collection/teacher_confidence_mean")),
                    _format_float(greedy.get("collection/disagreement_rate")),
                    _format_float(greedy.get("collection/route_entropy")),
                    _format_float(greedy.get("route_entropy")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", ""])
    if {7, 11, 19}.issubset(best_by_seed):
        best_11 = best_by_seed[11]
        best_7 = best_by_seed[7]
        best_19 = best_by_seed[19]
        lines.append(
            f"- Seed `11` is the successful case: best greedy success `{best_11['greedy_success']:.4f}` at round `{best_11['round']}`."
        )
        lines.append(
            f"- Seed `7` is partial: best greedy success `{best_7['greedy_success']:.4f}` at round `{best_7['round']}`."
        )
        lines.append(
            f"- Seed `19` is the failed case: best greedy success `{best_19['greedy_success']:.4f}`."
        )
        if (best_19["unique_ratio"] or 0.0) < (best_11["unique_ratio"] or 0.0):
            lines.append(
                f"- The failed seed shows lower learner-state coverage at its best round: unique-state ratio `{_format_float(best_19['unique_ratio'])}` vs `{_format_float(best_11['unique_ratio'])}` for the successful seed."
            )
        if (best_19["disagreement"] or 0.0) > (best_11["disagreement"] or 0.0):
            lines.append(
                f"- The failed seed also keeps higher teacher-student disagreement: `{_format_float(best_19['disagreement'])}` vs `{_format_float(best_11['disagreement'])}`."
            )
        if (best_19["teacher_confidence"] or 0.0) < (best_11["teacher_confidence"] or 0.0):
            lines.append(
                f"- Teacher labels on student-visited states are less confident in the failed seed: `{_format_float(best_19['teacher_confidence'])}` vs `{_format_float(best_11['teacher_confidence'])}`."
            )
        lines.append(
            f"- Seed `11` is the only seed whose winning round also shrinks the labeled dataset substantially (`6108` steps instead of `16000`), which points to stale append-all aggregation as a more plausible failure mode than weak teacher confidence."
        )
        lines.append(
            "- If none of the metrics above separate the seeds strongly, the remaining brittleness is likely optimization-sensitive rather than explained by a single obvious data-quality scalar."
        )
    return "\n".join(lines) + "\n"


def _build_sweep_report(rows: list[dict[str, Any]], token_rows: list[dict[str, Any]], episodes: int) -> str:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped[(str(row["method"]), int(row["seed"]))] = [*grouped.get((str(row["method"]), int(row["seed"])), []), row]
    lines = [
        "# Learner-State Robustness Sweep Report",
        "",
        f"- external evaluation episodes per mode: `{episodes}`",
        "",
        "| Method | Seed | Greedy Success | Best Sampled Success | Loss | Aggregation | Weighting | Unique Ratio | Disagreement | Teacher Conf |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    method_means: dict[str, list[float]] = {}
    for (method, seed), mode_rows in sorted(grouped.items()):
        greedy = _greedy_row(mode_rows)
        _best_mode, best_sampled = _best_sampled(mode_rows)
        method_means.setdefault(method, []).append(float(greedy.get("eval_success_rate", 0.0)))
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    str(seed),
                    _format_float(greedy.get("eval_success_rate")),
                    _format_float(best_sampled),
                    str(greedy.get("loss", "-")),
                    str(greedy.get("aggregation", "-")),
                    str(greedy.get("weighting", "-")),
                    _format_float(greedy.get("collection/unique_state_ratio")),
                    _format_float(greedy.get("collection/disagreement_rate")),
                    _format_float(greedy.get("collection/teacher_confidence_mean")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Minimal Tokenized Sanity Check", ""])
    if token_rows:
        grouped_token: dict[int, list[dict[str, Any]]] = {}
        for row in token_rows:
            grouped_token.setdefault(int(row["seed"]), []).append(row)
        lines.append("| Seed | Greedy Success | Best Sampled Success | Method |")
        lines.append("| --- | ---: | ---: | --- |")
        for seed, mode_rows in sorted(grouped_token.items()):
            greedy = _greedy_row(mode_rows)
            _best_mode, best_sampled = _best_sampled(mode_rows)
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(seed),
                        _format_float(greedy.get("eval_success_rate")),
                        _format_float(best_sampled),
                        str(greedy.get("method")),
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Interpretation", ""])
    if method_means:
        ordered = sorted(method_means.items(), key=lambda item: sum(item[1]) / max(len(item[1]), 1), reverse=True)
        best_method, best_scores = ordered[0]
        lines.append(
            f"- The strongest method in this bounded sweep is `{best_method}` with mean greedy success `{sum(best_scores) / max(len(best_scores), 1):.4f}` on the seeds it was run on."
        )
    lines.append(
        "- The key questions for this report are whether seed `19` rises above `0.0`, whether seed `11` stays strong, and whether the improvement also appears for the minimal tokenized sanity check."
    )
    return "\n".join(lines) + "\n"


def _build_multiseed_report(rows: list[dict[str, Any]], episodes: int) -> str:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped[(int(row["seed"]), str(row["label"]))] = [*grouped.get((int(row["seed"]), str(row["label"])), []), row]
    labels = [
        ("flat_dense", "flat_dense"),
        ("recovered_token_dense", "recovered token_dense"),
        ("baseline_sare", "baseline SARE"),
        ("improved_lss_sare", "improved learner-state SARE"),
    ]
    means: dict[str, list[float]] = {label: [] for label, _ in labels}
    lines = [
        "# Learner-State Robustness Multi-Seed Report",
        "",
        f"- external evaluation episodes per mode: `{episodes}`",
        "",
        "| Seed | flat_dense | recovered token_dense | baseline SARE | improved learner-state SARE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for seed in sorted({key[0] for key in grouped}):
        values = [str(seed)]
        for label, _display in labels:
            greedy = _greedy_row(grouped[(seed, label)])
            success = float(greedy.get("eval_success_rate", 0.0))
            means[label].append(success)
            values.append(_format_float(success))
        lines.append("| " + " | ".join(values) + " |")
    mean_token = sum(means["recovered_token_dense"]) / max(len(means["recovered_token_dense"]), 1)
    mean_improved = sum(means["improved_lss_sare"]) / max(len(means["improved_lss_sare"]), 1)
    zero_fail = any(score <= 0.0 for score in means["improved_lss_sare"])
    lines.extend(
        [
            "",
            "## Mean Greedy Success",
            "",
            "| Variant | Mean Greedy Success |",
            "| --- | ---: |",
            f"| flat_dense | `{sum(means['flat_dense']) / max(len(means['flat_dense']), 1):.4f}` |",
            f"| recovered token_dense | `{mean_token:.4f}` |",
            f"| baseline SARE | `{sum(means['baseline_sare']) / max(len(means['baseline_sare']), 1):.4f}` |",
            f"| improved learner-state SARE | `{mean_improved:.4f}` |",
            "",
            "## Verdict",
            "",
        ]
    )
    if mean_improved >= mean_token and not zero_fail:
        lines.append("- The improved learner-state method passes the repo’s reopen-routed-claim bar on the 3-seed external gate.")
    elif mean_improved > sum(means["baseline_sare"]) / max(len(means["baseline_sare"]), 1):
        lines.append("- The improved learner-state method beats baseline PPO SARE but still fails the repo’s reopen-routed-claim bar.")
    else:
        lines.append("- The improved learner-state method is negative or no better than the current learner-state baseline.")
    if zero_fail:
        lines.append("- At least one seed remains a complete greedy failure at `0.0`, so the hard robustness bar is not met.")
    return "\n".join(lines) + "\n"


def reproduction_note(args: argparse.Namespace) -> None:
    root = Path(args.root)
    rows = _evaluate_targets(_baseline_targets(root), args.device, args.episodes)
    _write_main_process_report(rows, args.output, args.csv, _build_reproduction_note(rows, root, args.episodes))


def heterogeneity_report(args: argparse.Namespace) -> None:
    rows = _evaluate_targets(_lss_round_targets(Path(args.root)), args.device, args.episodes)
    _write_main_process_report(rows, args.output, args.csv, _build_heterogeneity_report(rows, args.episodes))


def sweep_report(args: argparse.Namespace) -> None:
    all_paths = list(args.paths)
    if args.token_sanity_root:
        all_paths.append(args.token_sanity_root)
    all_rows = _evaluate_targets(_sweep_targets(all_paths), args.device, args.episodes)
    rows = [row for row in all_rows if row.get("variant") == "sare"]
    token_rows = [row for row in all_rows if row.get("variant") == "token_dense"]
    combined_rows = rows + token_rows
    _write_main_process_report(combined_rows, args.output, args.csv, _build_sweep_report(rows, token_rows, args.episodes))


def multiseed_report(args: argparse.Namespace) -> None:
    baseline_root = Path(args.baseline_root)
    improved_root = Path(args.improved_root)
    baseline_targets = _baseline_targets(baseline_root)
    improved_targets: list[EvalTarget] = []
    for target in _sweep_targets([str(improved_root)]):
        if target.variant != "sare":
            continue
        improved_targets.append(
            EvalTarget(
                seed=target.seed,
                label="improved_lss_sare",
                variant=target.variant,
                config_path=target.config_path,
                checkpoint_path=target.checkpoint_path,
                run_dir=target.run_dir,
                method=target.method,
                stage=target.stage,
                command_path=target.command_path,
            )
        )
    filtered_baselines = [
        target for target in baseline_targets
        if target.label in {"flat_dense", "recovered_token_dense", "baseline_sare"}
    ]
    rows = _evaluate_targets(filtered_baselines + improved_targets, args.device, args.episodes)
    _write_main_process_report(rows, args.output, args.csv, _build_multiseed_report(rows, args.episodes))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze learner-state supervision robustness experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reproduction = subparsers.add_parser("reproduction-note")
    reproduction.add_argument("--root", required=True)
    reproduction.add_argument("--episodes", type=int, default=64)
    reproduction.add_argument("--device", default="auto")
    reproduction.add_argument("--output", required=True)
    reproduction.add_argument("--csv", default=None)

    heterogeneity = subparsers.add_parser("heterogeneity-report")
    heterogeneity.add_argument("--root", required=True)
    heterogeneity.add_argument("--episodes", type=int, default=64)
    heterogeneity.add_argument("--device", default="auto")
    heterogeneity.add_argument("--output", required=True)
    heterogeneity.add_argument("--csv", default=None)

    sweep = subparsers.add_parser("sweep-report")
    sweep.add_argument("paths", nargs="+")
    sweep.add_argument("--token-sanity-root", default=None)
    sweep.add_argument("--episodes", type=int, default=64)
    sweep.add_argument("--device", default="auto")
    sweep.add_argument("--output", required=True)
    sweep.add_argument("--csv", default=None)

    multiseed = subparsers.add_parser("multiseed-report")
    multiseed.add_argument("--baseline-root", required=True)
    multiseed.add_argument("--improved-root", required=True)
    multiseed.add_argument("--episodes", type=int, default=64)
    multiseed.add_argument("--device", default="auto")
    multiseed.add_argument("--output", required=True)
    multiseed.add_argument("--csv", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "reproduction-note":
        reproduction_note(args)
        return
    if args.command == "heterogeneity-report":
        heterogeneity_report(args)
        return
    if args.command == "sweep-report":
        sweep_report(args)
        return
    multiseed_report(args)


if __name__ == "__main__":
    main()
