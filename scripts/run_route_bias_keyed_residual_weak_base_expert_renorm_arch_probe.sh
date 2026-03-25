#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

ROOT="outputs/diagnostics/architecture_route_bias_keyed_residual_weak_base_expert_renorm_probe"
REPORT_MD="outputs/reports/architecture_route_bias_keyed_residual_weak_base_expert_renorm_probe.md"
REPORT_JSON="outputs/reports/architecture_route_bias_keyed_residual_weak_base_expert_renorm_probe.json"
RUN_SUMMARY_MD="outputs/reports/architecture_route_bias_keyed_residual_weak_base_expert_renorm_probe_run_summary.md"
RUN_SUMMARY_CSV="outputs/reports/architecture_route_bias_keyed_residual_weak_base_expert_renorm_probe_run_summary.csv"
TMP_CFG_DIR="$ROOT/generated_configs"
rm -rf "$ROOT"
mkdir -p "$ROOT" "$TMP_CFG_DIR" "$(dirname "$REPORT_MD")"

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
else
  GPU_COUNT=1
fi
if [[ -z "$GPU_COUNT" || "$GPU_COUNT" -lt 1 ]]; then
  GPU_COUNT=1
fi

declare -a CONFIGS=(
  "configs/diagnostic/minigrid_doorkey_sare_phase_memory_probe.yaml"
  "configs/diagnostic/minigrid_doorkey_sare_phase_memory_route_bias_probe.yaml"
  "configs/diagnostic/minigrid_doorkey_sare_phase_memory_route_bias_keyed_residual_probe.yaml"
  "configs/diagnostic/minigrid_doorkey_sare_phase_memory_route_bias_keyed_residual_weak_base_expert_renorm_probe.yaml"
)
declare -a SEEDS=(7 11)

declare -a AVAILABLE_GPUS=()
for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
  AVAILABLE_GPUS+=("$gpu")
done
declare -a ACTIVE_PIDS=()
declare -a ACTIVE_GPUS=()
declare -a ACTIVE_RUN_DIRS=()

update_manifest_end() {
  local run_dir="$1"
  local exit_code="$2"
  ./.venv/bin/python - "$run_dir" "$exit_code" <<'PY'
import json
import sys
import time
from pathlib import Path

run_dir = Path(sys.argv[1])
exit_code = int(sys.argv[2])
manifest_path = run_dir / "run_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["ended_at"] = time.time()
manifest["exit_code"] = exit_code
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
PY
}

reclaim_one_gpu() {
  while true; do
    for i in "${!ACTIVE_PIDS[@]}"; do
      local pid="${ACTIVE_PIDS[$i]}"
      if ! kill -0 "$pid" 2>/dev/null; then
        local status=0
        if wait "$pid"; then
          status=0
        else
          status=$?
        fi
        local gpu="${ACTIVE_GPUS[$i]}"
        local run_dir="${ACTIVE_RUN_DIRS[$i]}"
        update_manifest_end "$run_dir" "$status"
        unset 'ACTIVE_PIDS[i]'
        unset 'ACTIVE_GPUS[i]'
        unset 'ACTIVE_RUN_DIRS[i]'
        ACTIVE_PIDS=("${ACTIVE_PIDS[@]}")
        ACTIVE_GPUS=("${ACTIVE_GPUS[@]}")
        ACTIVE_RUN_DIRS=("${ACTIVE_RUN_DIRS[@]}")
        AVAILABLE_GPUS+=("$gpu")
        if [[ "$status" -ne 0 ]]; then
          exit "$status"
        fi
        return
      fi
    done
    sleep 1
  done
}

for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    while [[ "${#AVAILABLE_GPUS[@]}" -eq 0 ]]; do
      reclaim_one_gpu
    done
    gpu="${AVAILABLE_GPUS[0]}"
    AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]:1}")
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
    "family": "architecture_route_bias_keyed_residual_weak_base_expert_renorm_probe",
    "started_at": time.time(),
}
(run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
PY
    CUDA_VISIBLE_DEVICES="$gpu" torchrun --standalone --nproc_per_node=1 -m psmn_rl.launch --config "$generated_cfg" &
    ACTIVE_PIDS+=("$!")
    ACTIVE_GPUS+=("$gpu")
    ACTIVE_RUN_DIRS+=("$run_dir")
  done
done

while [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
  reclaim_one_gpu
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
    if "sare_phase_memory_route_bias_keyed_residual_weak_base_expert_renorm" in config_text:
        variant = "sare_phase_memory_route_bias_keyed_residual_weak_base_expert_renorm"
    elif "sare_phase_memory_route_bias_keyed_residual" in config_text:
        variant = "sare_phase_memory_route_bias_keyed_residual"
    elif "sare_phase_memory_route_bias" in config_text:
        variant = "sare_phase_memory_route_bias"
    else:
        variant = "sare_phase_memory"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    last = None
    best_train_return = float("-inf")
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("type") == "scalar":
                last = payload
                best_train_return = max(best_train_return, float(payload.get("train/episode_return", 0.0)))
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
            "memory_route_bias_norm": float(last.get("memory/route_bias_norm", 0.0)),
            "memory_base_route_bias_logits_norm": float(last.get("memory/base_route_bias_logits_norm", 0.0)),
            "memory_keyed_route_bias_logits_norm": float(last.get("memory/keyed_route_bias_logits_norm", 0.0)),
            "memory_reweighted_keyed_route_bias_logits_norm": float(last.get("memory/reweighted_keyed_route_bias_logits_norm", 0.0)),
            "memory_renormed_keyed_route_bias_logits_norm": float(last.get("memory/renormed_keyed_route_bias_logits_norm", 0.0)),
            "memory_weak_base_expert_gate_mean": float(last.get("memory/weak_base_expert_gate_mean", 0.0)),
            "memory_weak_base_expert_gate_std": float(last.get("memory/weak_base_expert_gate_std", 0.0)),
            "memory_keyed_residual_scale_mean": float(last.get("memory/keyed_residual_scale_mean", 0.0)),
            "memory_keyed_residual_norm_preserve_scale_mean": float(last.get("memory/keyed_residual_norm_preserve_scale_mean", 0.0)),
            "memory_route_bias_logits_norm": float(last.get("memory/route_bias_logits_norm", 0.0)),
            "best_train_return": best_train_return if best_train_return != float("-inf") else 0.0,
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
        "memory_route_bias_norm",
        "memory_base_route_bias_logits_norm",
        "memory_keyed_route_bias_logits_norm",
        "memory_reweighted_keyed_route_bias_logits_norm",
        "memory_renormed_keyed_route_bias_logits_norm",
        "memory_weak_base_expert_gate_mean",
        "memory_weak_base_expert_gate_std",
        "memory_keyed_residual_scale_mean",
        "memory_keyed_residual_norm_preserve_scale_mean",
        "memory_route_bias_logits_norm",
        "best_train_return",
    ]
    out = {"runs": float(len(subset))}
    for key in keys:
        out[key] = statistics.fmean(row[key] for row in subset)
    return out

variants = {
    "sare_phase_memory": summarize("sare_phase_memory"),
    "sare_phase_memory_route_bias": summarize("sare_phase_memory_route_bias"),
    "sare_phase_memory_route_bias_keyed_residual": summarize("sare_phase_memory_route_bias_keyed_residual"),
    "sare_phase_memory_route_bias_keyed_residual_weak_base_expert_renorm": summarize("sare_phase_memory_route_bias_keyed_residual_weak_base_expert_renorm"),
}

report_json.write_text(json.dumps({"rows": rows, "variants": variants}, indent=2, sort_keys=True), encoding="utf-8")

lines = [
    "# Norm-Preserving Weak-Base Expert Renorm Keyed-Residual Probe",
    "",
    "| Variant | Runs | Eval Success | Eval Return | Train Success | Best Train Return | Throughput | Route Entropy | Route Bias Norm | Base Logits Norm | Keyed Logits Norm | Reweighted Keyed Norm | Renormed Keyed Norm | Weak-Base Gate Mean | Weak-Base Gate Std | Keyed Scale Mean | Renorm Scale Mean | Total Logits Norm |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for label, stats in variants.items():
    lines.append(
        "| {label} | {runs:.0f} | {eval_success_rate:.3f} | {eval_return:.3f} | {train_success_rate:.3f} | {best_train_return:.3f} | {throughput_fps:.1f} | {route_entropy:.3f} | {memory_route_bias_norm:.3f} | {memory_base_route_bias_logits_norm:.3f} | {memory_keyed_route_bias_logits_norm:.3f} | {memory_reweighted_keyed_route_bias_logits_norm:.3f} | {memory_renormed_keyed_route_bias_logits_norm:.3f} | {memory_weak_base_expert_gate_mean:.3f} | {memory_weak_base_expert_gate_std:.3f} | {memory_keyed_residual_scale_mean:.3f} | {memory_keyed_residual_norm_preserve_scale_mean:.3f} | {memory_route_bias_logits_norm:.3f} |".format(
            label=label,
            **stats,
        )
    )
report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
