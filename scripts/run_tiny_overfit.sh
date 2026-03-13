#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
OUTPUT_ROOT="${PSMN_OUTPUT_ROOT:-outputs/diagnostics/tiny_overfit}"
UPDATES="${PSMN_MAX_UPDATES:-}"

mkdir -p "$OUTPUT_ROOT"

CONFIGS=(
  "configs/diagnostic/minigrid_empty5_flat_dense_overfit.yaml"
  "configs/diagnostic/minigrid_empty5_token_dense_overfit.yaml"
)

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
  python -m psmn_rl.train "${args[@]}"
done

python -m psmn_rl.analysis.summarize \
  "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/report.md" \
  --csv "$OUTPUT_ROOT/report.csv" | tee "$OUTPUT_ROOT/summary.md"
