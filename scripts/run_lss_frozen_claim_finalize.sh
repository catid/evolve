#!/usr/bin/env bash
set -euo pipefail

EPISODES="${PSMN_EVAL_EPISODES:-64}"
CURRENT_CSV="${PSMN_CURRENT_CSV:-outputs/reports/lss_fresh_single_expert_matched_control_report.csv}"
FINAL_SINGLE_EXPERT_CSV="${PSMN_FINAL_SINGLE_EXPERT_CSV:-outputs/reports/lss_final_block_single_expert_control_report.csv}"
COMBINED_OUTPUT="${PSMN_COMBINED_OUTPUT:-outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.md}"
COMBINED_CSV="${PSMN_COMBINED_CSV:-outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.csv}"
FAILURE_CSV="${PSMN_FAILURE_CSV:-outputs/reports/lss_final_block_failure_analysis.csv}"
MEMO_OUTPUT="${PSMN_MEMO_OUTPUT:-outputs/reports/lss_frozen_claim_decision_memo.md}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_frozen_claim updated-combined-doorkey-report \
  --current-csv "$CURRENT_CSV" \
  --final-single-expert-csv "$FINAL_SINGLE_EXPERT_CSV" \
  --episodes "$EPISODES" \
  --output "$COMBINED_OUTPUT" \
  --csv "$COMBINED_CSV"

python -m psmn_rl.analysis.lss_frozen_claim decision-memo \
  --final-single-expert-csv "$FINAL_SINGLE_EXPERT_CSV" \
  --combined-csv "$COMBINED_CSV" \
  --failure-csv "$FAILURE_CSV" \
  --output "$MEMO_OUTPUT"
