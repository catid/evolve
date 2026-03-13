from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml


def load_latest_metrics(metrics_path: Path) -> dict[str, float]:
    latest: dict[str, float] = {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("type") != "scalar":
                continue
            for key, value in payload.items():
                if key not in {"type", "step"}:
                    latest[key] = float(value)
    return latest


def infer_group(run_name: str) -> tuple[str, str]:
    if run_name.startswith("baseline_"):
        remainder = run_name[len("baseline_") :]
        for suffix in ("_flat_dense", "_dense", "_single_expert"):
            if remainder.endswith(suffix):
                return remainder[: -len(suffix)], suffix[1:]
    if run_name.startswith("sare_"):
        return run_name[len("sare_") :], "sare"
    if run_name.startswith("treg_h_"):
        return run_name[len("treg_h_") :], "treg_h"
    if run_name.startswith("srw_"):
        return run_name[len("srw_") :], "srw"
    if run_name.startswith("por_"):
        return run_name[len("por_") :], "por"
    return run_name, "unknown"


def infer_from_config(run_dir: Path) -> tuple[str, str] | None:
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.exists():
        return None
    payload = yaml.safe_load(config_path.read_text()) or {}
    env_id = payload.get("env", {}).get("env_id")
    variant = payload.get("model", {}).get("variant")
    if not env_id or not variant:
        return None
    env_name = str(env_id).replace("MiniGrid-", "").replace("procgen_gym/procgen-", "").replace("-v0", "")
    env_name = env_name.replace("Dynamic-Obstacles-", "dynamic_obstacles_").replace("-", "_").lower()
    return env_name, str(variant)


def build_report(root: Path) -> str:
    grouped: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for metrics_path in sorted(root.glob("*/metrics.jsonl")):
        run_name = metrics_path.parent.name
        inferred = infer_from_config(metrics_path.parent)
        if inferred is not None:
            env_name, variant_name = inferred
        else:
            env_name, variant_name = infer_group(run_name)
        grouped[env_name][variant_name] = load_latest_metrics(metrics_path)

    lines = ["# Variant Comparison", ""]
    for env_name in sorted(grouped):
        lines.append(f"## {env_name}")
        lines.append("")
        lines.append("| variant | eval_return | success | throughput | compute_proxy | route_entropy |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for variant_name in sorted(grouped[env_name]):
            metrics = grouped[env_name][variant_name]
            lines.append(
                "| {variant} | {eval_return:.3f} | {success:.3f} | {throughput:.1f} | {compute:.3f} | {entropy:.3f} |".format(
                    variant=variant_name,
                    eval_return=metrics.get("eval_return", 0.0),
                    success=metrics.get("eval_success_rate", 0.0),
                    throughput=metrics.get("throughput_fps", 0.0),
                    compute=metrics.get("active_compute_proxy", 0.0),
                    entropy=metrics.get("route_entropy", 0.0),
                )
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ablation runs grouped by inferred environment name.")
    parser.add_argument("root", help="Root ablation directory containing per-run folders")
    parser.add_argument("--output", default=None, help="Optional output markdown path")
    args = parser.parse_args()

    root = Path(args.root)
    report = build_report(root)
    if args.output is not None:
        Path(args.output).write_text(report)
    print(report)


if __name__ == "__main__":
    main()
