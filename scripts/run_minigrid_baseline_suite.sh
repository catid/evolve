#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
OUTPUT_ROOT="${PSMN_OUTPUT_ROOT:-outputs/experiments/minigrid_baselines}"
UPDATES="${PSMN_MAX_UPDATES:-}"

mkdir -p "$OUTPUT_ROOT"

CONFIGS=(
  "configs/experiments/minigrid_doorkey_flat_dense.yaml"
  "configs/experiments/minigrid_doorkey_token_dense.yaml"
  "configs/experiments/minigrid_doorkey_single_expert.yaml"
  "configs/experiments/minigrid_keycorridor_flat_dense.yaml"
  "configs/experiments/minigrid_keycorridor_token_dense.yaml"
  "configs/experiments/minigrid_keycorridor_single_expert.yaml"
  "configs/experiments/minigrid_dynamic_obstacles_flat_dense.yaml"
  "configs/experiments/minigrid_dynamic_obstacles_token_dense.yaml"
  "configs/experiments/minigrid_dynamic_obstacles_single_expert.yaml"
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
  args=(
    --config "$config"
    --device "$DEVICE"
    --output-dir "$OUTPUT_ROOT/$run_name"
  )
  if [[ -n "$UPDATES" ]]; then
    args+=(--max-updates "$UPDATES")
  fi
  "${LAUNCHER[@]}" "${args[@]}"
done

python -m psmn_rl.analysis.summarize \
  "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/report.md" \
  --csv "$OUTPUT_ROOT/report.csv" | tee "$OUTPUT_ROOT/summary.md"
