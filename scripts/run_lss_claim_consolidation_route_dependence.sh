#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
ORIGINAL_RUN="${PSMN_ORIGINAL_RUN:-outputs/reproductions/lss_claim_hardening_baseline/seed_7/kl_lss_sare}"
FRESH_RUN="${PSMN_FRESH_RUN:-outputs/experiments/lss_claim_hardening/additional_seeds/seed_23/kl_lss_sare}"
OUTPUT_PATH="${PSMN_REPORT_OUTPUT:-outputs/reports/lss_causal_route_dependence_report.md}"
CSV_PATH="${PSMN_REPORT_CSV:-outputs/reports/lss_causal_route_dependence_report.csv}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_route_dependence \
  --original-run "$ORIGINAL_RUN" \
  --fresh-run "$FRESH_RUN" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output "$OUTPUT_PATH" \
  --csv "$CSV_PATH"
