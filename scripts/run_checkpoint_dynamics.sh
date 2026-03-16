#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
OUTPUT_ROOT="${PSMN_OUTPUT_ROOT:-outputs/experiments/greedy_recovery/checkpoint_dynamics}"
UPDATES="${PSMN_MAX_UPDATES:-}"
EPISODES="${PSMN_EVAL_EPISODES:-32}"

mkdir -p "$OUTPUT_ROOT"

CONFIGS=(
  "configs/experiments/minigrid_doorkey_single_expert_ent1e3_ckptscan.yaml"
  "configs/experiments/minigrid_doorkey_sare_ent1e3_ckptscan.yaml"
)

if [[ "$DEVICE" == "cpu" ]]; then
  LAUNCHER=(python -m psmn_rl.train)
else
  NPROC_PER_NODE=$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)
  LAUNCHER=(torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m psmn_rl.launch)
fi

for config in "${CONFIGS[@]}"; do
  run_name="$(basename "${config%.yaml}")"
  cmd=("${LAUNCHER[@]}" --config "$config" --device "$DEVICE" --output-dir "$OUTPUT_ROOT/$run_name")
  if [[ -n "$UPDATES" ]]; then
    cmd+=(--max-updates "$UPDATES")
  fi
  "${cmd[@]}"
done

python -m psmn_rl.analysis.checkpoint_dynamics \
  "$OUTPUT_ROOT" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output outputs/reports/checkpoint_dynamics_report.md \
  --csv outputs/reports/checkpoint_dynamics_report.csv
