#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

ROOT="outputs/diagnostics/architecture_route_bias_keyed_residual_topk_masked_probe"
REPORT_MD="outputs/reports/architecture_route_bias_keyed_residual_topk_masked_probe.md"
REPORT_JSON="outputs/reports/architecture_route_bias_keyed_residual_topk_masked_probe.json"
RUN_SUMMARY_MD="outputs/reports/architecture_route_bias_keyed_residual_topk_masked_probe_run_summary.md"
RUN_SUMMARY_CSV="outputs/reports/architecture_route_bias_keyed_residual_topk_masked_probe_run_summary.csv"
TMP_CFG_DIR="$ROOT/generated_configs"
rm -rf "$ROOT"
mkdir -p "$ROOT" "$TMP_CFG_DIR" "$(dirname "$REPORT_MD")"

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
if [[ -z "$GPU_COUNT" || "$GPU_COUNT" -lt 1 ]]; then
  GPU_COUNT=1
fi

declare -a CONFIGS=(
  "configs/diagnostic/minigrid_doorkey_sare_phase_memory_probe.yaml"
  "configs/diagnostic/minigrid_doorkey_sare_phase_memory_route_bias_probe.yaml"
  "configs/diagnostic/minigrid_doorkey_sare_phase_memory_route_bias_keyed_residual_probe.yaml"
  "configs/diagnostic/minigrid_doorkey_sare_phase_memory_route_bias_keyed_residual_topk_masked_probe.yaml"
)
declare -a SEEDS=(7 11)

jobs=()
slot=0
for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    gpu=$((slot % GPU_COUNT))
    slot=$((slot + 1))
    generated_cfg="$TMP_CFG_DIR/$(basename "${cfg%.yaml}")_seed${seed}.yaml"
    ./.venv/bin/python - "$cfg" "$generated_cfg" "$seed" "$ROOT" <<'PY'
import sys
from pathlib import Path
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
seed = int(sys.argv[3])
root = Path(sys.argv[4])
payload = yaml.safe_load(src.read_text()) or {}
run_name = str(payload["logging"]["run_name"])
payload["seed"] = seed
payload["logging"]["run_name"] = f"{run_name}_seed{seed}"
payload["logging"]["output_dir"] = str(root / f"{run_name}_seed{seed}")
dst.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY
    run_dir=$(.venv/bin/python - "$generated_cfg" <<'PY'
from pathlib import Path
from psmn_rl.config import load_config
import sys

cfg = load_config(Path(sys.argv[1]))
print(cfg.logging.output_dir)
PY
)
    ./.venv/bin/python - "$generated_cfg" "$run_dir" "$gpu" <<'PY'
import hashlib
import json
import sys
import time
from pathlib import Path

cfg_path = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
gpu = int(sys.argv[3])
run_dir.mkdir(parents=True, exist_ok=True)
manifest = {
    "config_path": str(cfg_path),
    "config_hash": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
    "seed": int(cfg_path.stem.split("seed")[-1]),
    "gpu": gpu,
    "output_dir": str(run_dir),
    "family": "architecture_route_bias_keyed_residual_topk_masked_probe",
    "started_at": time.time(),
}
(run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
PY
    CUDA_VISIBLE_DEVICES="$gpu" torchrun --standalone --nproc_per_node=1 -m psmn_rl.launch --config "$generated_cfg" &
    jobs+=("$!")
  done
done

for job in "${jobs[@]}"; do
  wait "$job"
done

./.venv/bin/python -m psmn_rl.analysis.summarize \
  "$ROOT" \
  --output "$RUN_SUMMARY_MD" \
  --csv "$RUN_SUMMARY_CSV"

./.venv/bin/python - "$ROOT" "$REPORT_MD" "$REPORT_JSON" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
report_md = Path(sys.argv[2])
report_json = Path(sys.argv[3])

rows = []
for metrics_path in sorted(root.glob("*/metrics.jsonl")):
    run_dir = metrics_path.parent
    config_text = (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    if "sare_phase_memory_route_bias_keyed_residual_topk_masked" in config_text:
        variant = "sare_phase_memory_route_bias_keyed_residual_topk_masked"
    elif "sare_phase_memory_route_bias_keyed_residual" in config_text:
        variant = "sare_phase_memory_route_bias_keyed_residual"
    elif "sare_phase_memory_route_bias" in config_text:
        variant = "sare_phase_memory_route_bias"
    else:
        variant = "sare_phase_memory"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    last = None
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("type") == "scalar":
                last = payload
    if last is None:
        continue
    rows.append(
        {
            "run_dir": str(run_dir),
            "seed": manifest["seed"],
            "variant": variant,
            "eval_success_rate": float(last.get("eval_success_rate", 0.0)),
            "eval_return": float(last.get("eval_return", 0.0)),
            "train_success_rate": float(last.get("train/success_rate", 0.0)),
            "throughput_fps": float(last.get("throughput_fps", 0.0)),
            "route_entropy": float(last.get("route_entropy", 0.0)),
            "memory_mix": float(last.get("memory/mix", 0.0)),
            "memory_route_bias_norm": float(last.get("memory/route_bias_norm", 0.0)),
            "memory_base_route_bias_logits_norm": float(last.get("memory/base_route_bias_logits_norm", 0.0)),
            "memory_keyed_route_bias_logits_norm": float(last.get("memory/keyed_route_bias_logits_norm", 0.0)),
            "memory_masked_keyed_route_bias_logits_norm": float(last.get("memory/masked_keyed_route_bias_logits_norm", 0.0)),
            "memory_masked_keyed_route_bias_density": float(last.get("memory/masked_keyed_route_bias_density", 0.0)),
            "memory_route_bias_logits_norm": float(last.get("memory/route_bias_logits_norm", 0.0)),
        }
    )

def summarize(label: str) -> dict[str, float]:
    subset = [row for row in rows if row["variant"] == label]
    if not subset:
        return {}
    keys = [
        "eval_success_rate",
        "eval_return",
        "train_success_rate",
        "throughput_fps",
        "route_entropy",
        "memory_mix",
        "memory_route_bias_norm",
        "memory_base_route_bias_logits_norm",
        "memory_keyed_route_bias_logits_norm",
        "memory_masked_keyed_route_bias_logits_norm",
        "memory_masked_keyed_route_bias_density",
        "memory_route_bias_logits_norm",
    ]
    out = {"runs": float(len(subset))}
    for key in keys:
        out[key] = statistics.fmean(row[key] for row in subset)
    return out

variants = {
    "sare_phase_memory": summarize("sare_phase_memory"),
    "sare_phase_memory_route_bias": summarize("sare_phase_memory_route_bias"),
    "sare_phase_memory_route_bias_keyed_residual": summarize("sare_phase_memory_route_bias_keyed_residual"),
    "sare_phase_memory_route_bias_keyed_residual_topk_masked": summarize("sare_phase_memory_route_bias_keyed_residual_topk_masked"),
}
report_json.write_text(json.dumps({"rows": rows, "variants": variants}, indent=2, sort_keys=True), encoding="utf-8")

lines = [
    "# Top-K-Masked Keyed-Residual Route-Bias Architecture Probe",
    "",
    "- variants: `sare_phase_memory`, `sare_phase_memory_route_bias`, `sare_phase_memory_route_bias_keyed_residual`, `sare_phase_memory_route_bias_keyed_residual_topk_masked`",
    "- env: `MiniGrid-DoorKey-5x5-v0`",
    "- updates per run: `24`",
    "- seeds per variant: `2`",
    "",
    "## Aggregate",
    "",
    "| Variant | Runs | Eval Success | Eval Return | Train Success | Throughput | Route Entropy | Route Bias Norm | Base Logits Norm | Keyed Logits Norm | Masked Keyed Norm | Masked Density | Total Logits Norm |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for label in (
    "sare_phase_memory",
    "sare_phase_memory_route_bias",
    "sare_phase_memory_route_bias_keyed_residual",
    "sare_phase_memory_route_bias_keyed_residual_topk_masked",
):
    stats = variants[label]
    if not stats:
        continue
    lines.append(
        f"| `{label}` | `{stats['runs']:.0f}` | `{stats['eval_success_rate']:.3f}` | `{stats['eval_return']:.3f}` | `{stats['train_success_rate']:.3f}` | `{stats['throughput_fps']:.1f}` | `{stats['route_entropy']:.3f}` | `{stats['memory_route_bias_norm']:.3f}` | `{stats['memory_base_route_bias_logits_norm']:.3f}` | `{stats['memory_keyed_route_bias_logits_norm']:.3f}` | `{stats['memory_masked_keyed_route_bias_logits_norm']:.3f}` | `{stats['memory_masked_keyed_route_bias_density']:.3f}` | `{stats['memory_route_bias_logits_norm']:.3f}` |"
    )
lines.extend(
    [
        "",
        "## Per-Run",
        "",
        "| Variant | Seed | Eval Success | Eval Return | Train Success | Throughput | Route Bias Norm | Keyed Logits Norm | Masked Keyed Norm | Masked Density | Total Logits Norm | Run Dir |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
)
for row in rows:
    lines.append(
        f"| `{row['variant']}` | `{row['seed']}` | `{row['eval_success_rate']:.3f}` | `{row['eval_return']:.3f}` | `{row['train_success_rate']:.3f}` | `{row['throughput_fps']:.1f}` | `{row['memory_route_bias_norm']:.3f}` | `{row['memory_keyed_route_bias_logits_norm']:.3f}` | `{row['memory_masked_keyed_route_bias_logits_norm']:.3f}` | `{row['memory_masked_keyed_route_bias_density']:.3f}` | `{row['memory_route_bias_logits_norm']:.3f}` | `{row['run_dir']}` |"
    )
report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
