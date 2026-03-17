#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
STRUCTURED_CSV="${PSMN_STRUCTURED_CSV:-outputs/reports/lss_fresh_single_expert_matched_control_report.csv}"
FINAL_CSV="${PSMN_FINAL_CSV:-outputs/reports/lss_resume_gate_reproduction_note.csv}"
PRIOR_FAILURE_CSV="${PSMN_PRIOR_FAILURE_CSV:-outputs/reports/lss_final_block_failure_analysis.csv}"
OUTPUT_PATH="${PSMN_REPORT_OUTPUT:-outputs/reports/lss_resume_gate_failure_mechanism_report.md}"
CSV_PATH="${PSMN_REPORT_CSV:-outputs/reports/lss_resume_gate_failure_mechanism_report.csv}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_resume_gate failure-mechanism-report \
  --structured-csv "$STRUCTURED_CSV" \
  --final-csv "$FINAL_CSV" \
  --prior-failure-csv "$PRIOR_FAILURE_CSV" \
  --device "$DEVICE" \
  --output "$OUTPUT_PATH" \
  --csv "$CSV_PATH"
