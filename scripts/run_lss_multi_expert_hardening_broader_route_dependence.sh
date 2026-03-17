#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
EXTRA_OUTPUT="${PSMN_EXTRA_OUTPUT:-outputs/reports/lss_multi_expert_hardening_route_dependence_extra.md}"
EXTRA_CSV="${PSMN_EXTRA_CSV:-outputs/reports/lss_multi_expert_hardening_route_dependence_extra.csv}"
OUTPUT_PATH="${PSMN_REPORT_OUTPUT:-outputs/reports/lss_broader_route_dependence_report.md}"
CSV_PATH="${PSMN_REPORT_CSV:-outputs/reports/lss_broader_route_dependence_report.csv}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_route_dependence \
  --case fresh 31 outputs/experiments/lss_claim_hardening/additional_seeds/seed_31/kl_lss_sare \
  --case fresh_extra 37 outputs/experiments/lss_claim_broadening/additional_fresh_block/seed_37/kl_lss_sare \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output "$EXTRA_OUTPUT" \
  --csv "$EXTRA_CSV"

python -m psmn_rl.analysis.lss_multi_expert_hardening broader-route-dependence-report \
  --existing-csv outputs/reports/lss_extended_route_dependence_report.csv \
  --extra-csv "$EXTRA_CSV" \
  --forensics-csv outputs/reports/lss_seed29_route_randomization_forensics.csv \
  --episodes "$EPISODES" \
  --output "$OUTPUT_PATH" \
  --csv "$CSV_PATH"
