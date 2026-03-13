#!/usr/bin/env bash
set -euo pipefail

UPDATES="${PSMN_MAX_UPDATES:-2}"
DEVICE="${PSMN_DEVICE:-auto}"
OUTPUT_ROOT="${PSMN_OUTPUT_ROOT:-outputs/ablations/minigrid_suite}"

mkdir -p "$OUTPUT_ROOT"

CONFIGS=(
  "configs/baseline/minigrid_flat_dense.yaml"
  "configs/baseline/minigrid_dense.yaml"
  "configs/baseline/minigrid_single_expert.yaml"
  "configs/sare/minigrid_doorkey.yaml"
  "configs/baseline/minigrid_keycorridor_flat_dense.yaml"
  "configs/baseline/minigrid_keycorridor_dense.yaml"
  "configs/baseline/minigrid_keycorridor_single_expert.yaml"
  "configs/sare/minigrid_keycorridor.yaml"
  "configs/baseline/minigrid_memory_flat_dense.yaml"
  "configs/baseline/minigrid_memory_dense.yaml"
  "configs/baseline/minigrid_memory_single_expert.yaml"
  "configs/sare/minigrid_memory.yaml"
  "configs/baseline/minigrid_dynamic_obstacles_flat_dense.yaml"
  "configs/baseline/minigrid_dynamic_obstacles_dense.yaml"
  "configs/baseline/minigrid_dynamic_obstacles_single_expert.yaml"
  "configs/sare/minigrid_dynamic_obstacles.yaml"
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
  run_name="$(basename "$(dirname "$config")")_$(basename "${config%.yaml}")"
  "${LAUNCHER[@]}" \
    --config "$config" \
    --device "$DEVICE" \
    --max-updates "$UPDATES" \
    --output-dir "$OUTPUT_ROOT/$run_name"
done

python -m psmn_rl.analysis.summarize "$OUTPUT_ROOT"/*/metrics.jsonl | tee "$OUTPUT_ROOT/summary.md"
