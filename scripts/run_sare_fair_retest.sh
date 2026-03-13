#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
OUTPUT_ROOT="${PSMN_OUTPUT_ROOT:-outputs/experiments/sare_retest}"
UPDATES="${PSMN_MAX_UPDATES:-}"
EPISODES="${PSMN_EVAL_EPISODES:-32}"
TRACE_LIMIT="${PSMN_TRACE_LIMIT:-2}"

mkdir -p "$OUTPUT_ROOT"

CONFIGS=(
  "configs/experiments/minigrid_doorkey_flat_dense_ent1e3.yaml"
  "configs/experiments/minigrid_doorkey_token_dense_ent1e3.yaml"
  "configs/experiments/minigrid_doorkey_single_expert_ent1e3.yaml"
  "configs/experiments/minigrid_doorkey_sare_ent1e3.yaml"
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

python -m psmn_rl.analysis.policy_diagnostics \
  "$OUTPUT_ROOT" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output "$OUTPUT_ROOT/report.md" \
  --csv "$OUTPUT_ROOT/report.csv" \
  --trace-dir outputs/reports/sare_retest_traces \
  --trace-limit "$TRACE_LIMIT"
