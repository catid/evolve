#!/usr/bin/env bash
set -euo pipefail

CASEBOOK_CSV="${PSMN_CASEBOOK_CSV:-outputs/reports/lss_forensic_casebook.csv}"
ROUND_AUDIT_CSV="${PSMN_ROUND_AUDIT_CSV:-outputs/reports/lss_forensic_round_audit.csv}"
ROUTE_LOCALITY_CSV="${PSMN_ROUTE_LOCALITY_CSV:-outputs/reports/lss_forensic_route_locality.csv}"
SCORECARD_OUTPUT="${PSMN_SCORECARD_OUTPUT:-outputs/reports/lss_resume_qualification_scorecard.md}"
SCORECARD_CSV="${PSMN_SCORECARD_CSV:-outputs/reports/lss_resume_qualification_scorecard.csv}"
MEMO_OUTPUT="${PSMN_MEMO_OUTPUT:-outputs/reports/lss_forensic_atlas_decision_memo.md}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_forensic_atlas resume-scorecard \
  --casebook-csv "$CASEBOOK_CSV" \
  --round-audit-csv "$ROUND_AUDIT_CSV" \
  --route-locality-csv "$ROUTE_LOCALITY_CSV" \
  --output "$SCORECARD_OUTPUT" \
  --csv "$SCORECARD_CSV"

python -m psmn_rl.analysis.lss_forensic_atlas decision-memo \
  --scorecard-csv "$SCORECARD_CSV" \
  --output "$MEMO_OUTPUT"
