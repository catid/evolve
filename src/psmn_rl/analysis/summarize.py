from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize_metrics(path: Path) -> dict[str, float]:
    latest: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("type") != "scalar":
                continue
            for key, value in payload.items():
                if key not in {"type", "step"}:
                    latest[key] = float(value)
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PSMN RL metrics JSONL files.")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    grouped = defaultdict(dict)
    for run in args.runs:
        run_path = Path(run)
        grouped[run_path.parent.name] = summarize_metrics(run_path)
    for name, metrics in grouped.items():
        print(f"## {name}")
        for key in sorted(metrics):
            print(f"- {key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
