#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
OUTPUT_PATH="${PSMN_REPORT_OUTPUT:-outputs/reports/lss_extended_route_dependence_report.md}"
CSV_PATH="${PSMN_REPORT_CSV:-outputs/reports/lss_extended_route_dependence_report.csv}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_route_dependence \
  --case original 7 outputs/reproductions/lss_claim_hardening_baseline/seed_7/kl_lss_sare \
  --case original 19 outputs/reproductions/lss_claim_hardening_baseline/seed_19/kl_lss_sare \
  --case fresh 23 outputs/experiments/lss_claim_hardening/additional_seeds/seed_23/kl_lss_sare \
  --case fresh 29 outputs/experiments/lss_claim_hardening/additional_seeds/seed_29/kl_lss_sare \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output "$OUTPUT_PATH" \
  --csv "$CSV_PATH"
