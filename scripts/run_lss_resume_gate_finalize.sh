#!/usr/bin/env bash
set -euo pipefail

REPRO_CSV="${PSMN_REPRO_CSV:-outputs/reports/lss_resume_gate_reproduction_note.csv}"
FAILURE_CSV="${PSMN_FAILURE_CSV:-outputs/reports/lss_resume_gate_failure_mechanism_report.csv}"
COMBINED_CSV="${PSMN_COMBINED_CSV:-outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.csv}"
OUTPUT_PATH="${PSMN_REPORT_OUTPUT:-outputs/reports/lss_resume_gate_decision_memo.md}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_resume_gate decision-memo \
  --reproduction-csv "$REPRO_CSV" \
  --failure-csv "$FAILURE_CSV" \
  --combined-csv "$COMBINED_CSV" \
  --output "$OUTPUT_PATH"
