#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
ORIGINAL_BASELINE_ROOT="${PSMN_ORIGINAL_BASELINE_ROOT:-outputs/reproductions/lss_robustness_baseline}"
ORIGINAL_IMPROVED_ROOT="${PSMN_ORIGINAL_IMPROVED_ROOT:-outputs/reproductions/lss_claim_hardening_baseline}"
FRESH_BASELINE_ROOT="${PSMN_FRESH_BASELINE_ROOT:-outputs/experiments/lss_claim_hardening/additional_seeds}"
FRESH_IMPROVED_ROOT="${PSMN_FRESH_IMPROVED_ROOT:-outputs/experiments/lss_claim_hardening/additional_seeds}"
OUTPUT_PATH="${PSMN_REPORT_OUTPUT:-outputs/reports/lss_claim_consolidation_reproduction_note.md}"
CSV_PATH="${PSMN_REPORT_CSV:-outputs/reports/lss_claim_consolidation_reproduction_note.csv}"
source .venv/bin/activate
python -m psmn_rl.analysis.lss_claim_consolidation reproduction-note \
  --original-baseline-root "$ORIGINAL_BASELINE_ROOT" \
  --original-improved-root "$ORIGINAL_IMPROVED_ROOT" \
  --fresh-baseline-root "$FRESH_BASELINE_ROOT" \
  --fresh-improved-root "$FRESH_IMPROVED_ROOT" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output "$OUTPUT_PATH" \
  --csv "$CSV_PATH"
