#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
OUTPUT_PATH="${PSMN_REPORT_OUTPUT:-outputs/reports/lss_multi_expert_hardening_reproduction_note.md}"
CSV_PATH="${PSMN_REPORT_CSV:-outputs/reports/lss_multi_expert_hardening_reproduction_note.csv}"

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
eval "${ANALYSIS_LAUNCHER} reproduction-note \
  --original-baseline-root outputs/reproductions/lss_robustness_baseline \
  --original-token-root outputs/experiments/lss_claim_hardening/matched_controls \
  --original-single-expert-root outputs/experiments/lss_claim_broadening/single_expert_controls \
  --original-sare-root outputs/reproductions/lss_claim_hardening_baseline \
  --fresh-baseline-root outputs/experiments/lss_claim_hardening/additional_seeds \
  --fresh-token-root outputs/experiments/lss_claim_consolidation/fresh_matched_controls \
  --fresh-sare-root outputs/experiments/lss_claim_hardening/additional_seeds \
  --fresh-extra-baseline-root outputs/experiments/lss_claim_broadening/additional_fresh_block \
  --fresh-extra-token-root outputs/experiments/lss_claim_broadening/additional_fresh_matched_controls \
  --fresh-extra-sare-root outputs/experiments/lss_claim_broadening/additional_fresh_block \
  --episodes \"$EPISODES\" \
  --device \"$DEVICE\" \
  --output \"$OUTPUT_PATH\" \
  --csv \"$CSV_PATH\""
