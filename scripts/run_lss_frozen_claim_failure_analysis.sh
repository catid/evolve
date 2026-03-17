#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
STRUCTURED_CSV="${PSMN_STRUCTURED_CSV:-outputs/reports/lss_fresh_single_expert_matched_control_report.csv}"
FINAL_CSV="${PSMN_FINAL_CSV:-outputs/reports/lss_final_fresh_seed_block_report.csv}"
FINAL_SINGLE_EXPERT_CSV="${PSMN_FINAL_SINGLE_EXPERT_CSV:-outputs/reports/lss_final_block_single_expert_control_report.csv}"
FINAL_SINGLE_EXPERT_ROOT="${PSMN_FINAL_SINGLE_EXPERT_ROOT:-outputs/experiments/lss_frozen_claim/final_block_single_expert_controls}"
OUTPUT_PATH="${PSMN_REPORT_OUTPUT:-outputs/reports/lss_final_block_failure_analysis.md}"
CSV_PATH="${PSMN_REPORT_CSV:-outputs/reports/lss_final_block_failure_analysis.csv}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_frozen_claim final-block-failure-analysis \
  --structured-csv "$STRUCTURED_CSV" \
  --final-csv "$FINAL_CSV" \
  --final-single-expert-csv "$FINAL_SINGLE_EXPERT_CSV" \
  --final-single-expert-root "$FINAL_SINGLE_EXPERT_ROOT" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output "$OUTPUT_PATH" \
  --csv "$CSV_PATH"
