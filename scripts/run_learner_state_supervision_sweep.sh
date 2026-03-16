#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
SPEC_ROOT="configs/experiments/teacher_extraction"
OUTPUT_ROOT="outputs/experiments/teacher_extraction/learner_state"
LOG_ROOT="${OUTPUT_ROOT}/logs"

mkdir -p "$LOG_ROOT"

SPECS=(
  "${SPEC_ROOT}/flat_dense_to_token_dense_lss.yaml"
  "${SPEC_ROOT}/flat_dense_to_sare_lss.yaml"
)

run_spec() {
  local spec_path="$1"
  local device_arg="$2"
  python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec_path" --device "$device_arg"
}

if [[ "$DEVICE" == "cpu" ]]; then
  for spec in "${SPECS[@]}"; do
    run_spec "$spec" cpu
  done
else
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_IDS <<<"${CUDA_VISIBLE_DEVICES}"
  else
    GPU_COUNT=$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)
    GPU_IDS=()
    for ((i = 0; i < GPU_COUNT; ++i)); do
      GPU_IDS+=("$i")
    done
  fi
  if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
    echo "No GPUs available for auto device runs" >&2
    exit 1
  fi
  slot=0
  for spec in "${SPECS[@]}"; do
    while (( $(jobs -rp | wc -l) >= ${#GPU_IDS[@]} )); do
      wait -n
    done
    gpu="${GPU_IDS[$slot]}"
    log_path="${LOG_ROOT}/$(basename "${spec%.yaml}").log"
    (CUDA_VISIBLE_DEVICES="$gpu" run_spec "$spec" auto >"$log_path" 2>&1) &
    slot=$(( (slot + 1) % ${#GPU_IDS[@]} ))
  done
  wait
fi

python -m psmn_rl.analysis.learner_state_supervision report \
  "$OUTPUT_ROOT" \
  --output outputs/reports/learner_state_supervision_report.md \
  --csv outputs/reports/learner_state_supervision_report.csv
