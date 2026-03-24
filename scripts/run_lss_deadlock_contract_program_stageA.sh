#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_deadlock_contract_program/campaign.yaml}"

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["family_definition"])
print(campaign["reports"]["teacher_audit_report"])
print(campaign["reports"]["teacher_audit_json"])
print(campaign["reports"]["distribution_audit_report"])
print(campaign["reports"]["distribution_audit_json"])
print(campaign["reports"]["shortlist_report"])
print(campaign["reports"]["shortlist_json"])
print(campaign["analysis"]["casebook_trace_episodes"])
print(campaign["analysis"]["casebook_max_steps"])
PY
)

FAMILY_OUTPUT="${cfg[0]}"
TEACHER_OUTPUT="${cfg[1]}"
TEACHER_JSON="${cfg[2]}"
DISTRIBUTION_OUTPUT="${cfg[3]}"
DISTRIBUTION_JSON="${cfg[4]}"
SHORTLIST_OUTPUT="${cfg[5]}"
SHORTLIST_JSON="${cfg[6]}"
TRACE_EPISODES="${cfg[7]}"
TRACE_MAX_STEPS="${cfg[8]}"

python -m psmn_rl.analysis.lss_deadlock_program family-definition \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$FAMILY_OUTPUT"

python -m psmn_rl.analysis.lss_deadlock_program teacher-audit \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$TEACHER_OUTPUT" \
  --json "$TEACHER_JSON" \
  --episodes "$TRACE_EPISODES" \
  --max-steps "$TRACE_MAX_STEPS"

python -m psmn_rl.analysis.lss_deadlock_program distribution-audit \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$DISTRIBUTION_OUTPUT" \
  --json "$DISTRIBUTION_JSON"

python -m psmn_rl.analysis.lss_deadlock_program shortlist \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$SHORTLIST_OUTPUT" \
  --json "$SHORTLIST_JSON"
