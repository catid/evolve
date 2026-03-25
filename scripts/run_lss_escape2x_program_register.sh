#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_escape2x_program/campaign.yaml}"

source .venv/bin/activate
readarray -t cfg < <(
  ./.venv/bin/python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["state_reconciliation"])
print(campaign["reports"]["baseline_sync"])
print(campaign["reports"]["baseline_sync_csv"])
print(campaign["reports"]["baseline_sync_json"])
PY
)

STATE_OUTPUT="${cfg[0]}"
BASELINE_OUTPUT="${cfg[1]}"
BASELINE_CSV="${cfg[2]}"
BASELINE_JSON="${cfg[3]}"

./.venv/bin/python -m psmn_rl.analysis.lss_escape2x state-reconciliation \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$STATE_OUTPUT"

./.venv/bin/python -m psmn_rl.analysis.lss_escape2x baseline-sync \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$BASELINE_OUTPUT" \
  --csv "$BASELINE_CSV" \
  --json "$BASELINE_JSON"
