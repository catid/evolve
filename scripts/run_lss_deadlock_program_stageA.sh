#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_deadlock_program/campaign.yaml}"

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["family_definition"])
print(campaign["reports"]["casebook_report"])
print(campaign["reports"]["casebook_csv"])
print(campaign["reports"]["casebook_json"])
print(campaign["reports"]["shortlist_report"])
print(campaign["reports"]["shortlist_json"])
print(campaign["analysis"]["casebook_trace_episodes"])
print(campaign["analysis"]["casebook_max_steps"])
PY
)

FAMILY_OUTPUT="${cfg[0]}"
CASEBOOK_OUTPUT="${cfg[1]}"
CASEBOOK_CSV="${cfg[2]}"
CASEBOOK_JSON="${cfg[3]}"
SHORTLIST_OUTPUT="${cfg[4]}"
SHORTLIST_JSON="${cfg[5]}"
TRACE_EPISODES="${cfg[6]}"
TRACE_MAX_STEPS="${cfg[7]}"

python -m psmn_rl.analysis.lss_deadlock_program family-definition \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$FAMILY_OUTPUT"

python -m psmn_rl.analysis.lss_deadlock_program casebook \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$CASEBOOK_OUTPUT" \
  --csv "$CASEBOOK_CSV" \
  --json "$CASEBOOK_JSON" \
  --episodes "$TRACE_EPISODES" \
  --max-steps "$TRACE_MAX_STEPS"

python -m psmn_rl.analysis.lss_deadlock_program shortlist \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$SHORTLIST_OUTPUT" \
  --json "$SHORTLIST_JSON"
