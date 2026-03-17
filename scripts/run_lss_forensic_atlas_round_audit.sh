#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CASE_CONFIG="${PSMN_CASE_CONFIG:-configs/experiments/lss_forensic_atlas/forensic_cases.yaml}"
MAX_SAMPLES="${PSMN_ROUTE_MAX_SAMPLES:-1024}"
OUTPUT_PATH="${PSMN_REPORT_OUTPUT:-outputs/reports/lss_forensic_round_audit.md}"
CSV_PATH="${PSMN_REPORT_CSV:-outputs/reports/lss_forensic_round_audit.csv}"

analysis_launcher() {
  source .venv/bin/activate
  local cuda_ok
  cuda_ok=$(python - <<'PY'
import torch
print(1 if torch.cuda.is_available() else 0)
PY
)
  if [[ "$DEVICE" == "cpu" || "$cuda_ok" -ne 1 ]]; then
    echo "python -m psmn_rl.analysis.lss_forensic_atlas"
    return
  fi
  local nproc
  nproc=$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)
  echo "torchrun --standalone --nproc_per_node=${nproc} -m psmn_rl.analysis.lss_forensic_atlas"
}

ANALYSIS_LAUNCHER="$(analysis_launcher)"
source .venv/bin/activate
eval "${ANALYSIS_LAUNCHER} round-audit \
  --case-config \"$CASE_CONFIG\" \
  --device \"$DEVICE\" \
  --max-samples \"$MAX_SAMPLES\" \
  --output \"$OUTPUT_PATH\" \
  --csv \"$CSV_PATH\""
