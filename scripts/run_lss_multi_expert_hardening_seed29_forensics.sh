#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
TRIAL_COUNT="${PSMN_TRIAL_COUNT:-8}"
OUTPUT_PATH="${PSMN_REPORT_OUTPUT:-outputs/reports/lss_seed29_route_randomization_forensics.md}"
CSV_PATH="${PSMN_REPORT_CSV:-outputs/reports/lss_seed29_route_randomization_forensics.csv}"

analysis_launcher() {
  source .venv/bin/activate
  local cuda_ok
  cuda_ok=$(python - <<'PY'
import torch
print(1 if torch.cuda.is_available() else 0)
PY
)
  if [[ "$DEVICE" == "cpu" || "$cuda_ok" -ne 1 ]]; then
    echo "python -m psmn_rl.analysis.lss_multi_expert_hardening"
    return
  fi
  local nproc
  nproc=$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)
  echo "torchrun --standalone --nproc_per_node=${nproc} -m psmn_rl.analysis.lss_multi_expert_hardening"
}

ANALYSIS_LAUNCHER="$(analysis_launcher)"
source .venv/bin/activate
eval "${ANALYSIS_LAUNCHER} seed29-route-randomization-forensics \
  --reference-csv outputs/reports/lss_extended_route_dependence_report.csv \
  --run-dir outputs/experiments/lss_claim_hardening/additional_seeds/seed_29/kl_lss_sare \
  --episodes \"$EPISODES\" \
  --trial-count \"$TRIAL_COUNT\" \
  --device \"$DEVICE\" \
  --output \"$OUTPUT_PATH\" \
  --csv \"$CSV_PATH\""
