#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
STRUCTURED_CSV="${PSMN_STRUCTURED_CSV:-outputs/reports/lss_fresh_single_expert_matched_control_report.csv}"
FINAL_CSV="${PSMN_FINAL_CSV:-outputs/reports/lss_final_fresh_seed_block_report.csv}"
OUTPUT_PATH="${PSMN_REPORT_OUTPUT:-outputs/reports/lss_frozen_claim_reproduction_note.md}"
CSV_PATH="${PSMN_REPORT_CSV:-outputs/reports/lss_frozen_claim_reproduction_note.csv}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_frozen_claim reproduction-note \
  --structured-csv "$STRUCTURED_CSV" \
  --final-csv "$FINAL_CSV" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output "$OUTPUT_PATH" \
  --csv "$CSV_PATH"
