#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/baseline/minigrid_dense.yaml}"
shift || true

NPROC_PER_NODE=$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m psmn_rl.launch \
  --config "$CONFIG_PATH" "$@"
