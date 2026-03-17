#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_TRACE_EPISODES:-64}"
MAX_STEPS="${PSMN_TRACE_MAX_STEPS:-256}"
CASE_CONFIG="${PSMN_CASE_CONFIG:-configs/experiments/lss_forensic_atlas/forensic_cases.yaml}"
OUTPUT_PATH="${PSMN_REPORT_OUTPUT:-outputs/reports/lss_forensic_casebook.md}"
CSV_PATH="${PSMN_REPORT_CSV:-outputs/reports/lss_forensic_casebook.csv}"
TRACE_JSON="${PSMN_TRACE_JSON:-outputs/reports/lss_forensic_casebook_traces.json}"

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
eval "${ANALYSIS_LAUNCHER} casebook \
  --case-config \"$CASE_CONFIG\" \
  --episodes \"$EPISODES\" \
  --max-steps \"$MAX_STEPS\" \
  --device \"$DEVICE\" \
  --output \"$OUTPUT_PATH\" \
  --csv \"$CSV_PATH\" \
  --trace-json \"$TRACE_JSON\""
