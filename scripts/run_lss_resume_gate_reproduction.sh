#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
FINAL_CSV="${PSMN_FINAL_CSV:-outputs/reports/lss_final_block_single_expert_control_report.csv}"
OUTPUT_PATH="${PSMN_REPORT_OUTPUT:-outputs/reports/lss_resume_gate_reproduction_note.md}"
CSV_PATH="${PSMN_REPORT_CSV:-outputs/reports/lss_resume_gate_reproduction_note.csv}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_resume_gate reproduction-note \
  --final-csv "$FINAL_CSV" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output "$OUTPUT_PATH" \
  --csv "$CSV_PATH"
