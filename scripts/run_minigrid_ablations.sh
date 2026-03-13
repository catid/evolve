#!/usr/bin/env bash
set -euo pipefail

UPDATES="${PSMN_MAX_UPDATES:-1}"
DEVICE="${PSMN_DEVICE:-auto}"
OUTPUT_ROOT="${PSMN_OUTPUT_ROOT:-outputs/ablations/minigrid_doorkey}"

mkdir -p "$OUTPUT_ROOT"

CONFIGS=(
  "configs/baseline/minigrid_flat_dense.yaml"
  "configs/baseline/minigrid_dense.yaml"
  "configs/baseline/minigrid_single_expert.yaml"
  "configs/sare/minigrid_doorkey.yaml"
  "configs/treg_h/minigrid_doorkey.yaml"
  "configs/srw/minigrid_doorkey.yaml"
  "configs/por/minigrid_doorkey.yaml"
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
