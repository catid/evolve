#!/usr/bin/env bash
set -euo pipefail

EPISODES="${PSMN_EVAL_EPISODES:-64}"
ORIGINAL_CSV="${PSMN_ORIGINAL_CSV:-outputs/reports/lss_matched_control_report.csv}"
FRESH_CSV="${PSMN_FRESH_CSV:-outputs/reports/lss_fresh_matched_control_report.csv}"
OUTPUT_PATH="${PSMN_REPORT_OUTPUT:-outputs/reports/lss_claim_broadening_reproduction_note.md}"
CSV_PATH="${PSMN_REPORT_CSV:-outputs/reports/lss_claim_broadening_reproduction_note.csv}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_claim_broadening reproduction-note \
  --original-csv "$ORIGINAL_CSV" \
  --fresh-csv "$FRESH_CSV" \
  --episodes "$EPISODES" \
  --output "$OUTPUT_PATH" \
  --csv "$CSV_PATH"
