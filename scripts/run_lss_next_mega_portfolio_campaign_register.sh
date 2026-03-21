#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_next_mega_portfolio_campaign/campaign.yaml}"

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["state_reconciliation"])
print(campaign["reports"]["registration"])
print(campaign["reports"]["baseline_sync"])
print(campaign["reports"]["baseline_sync_csv"])
print(campaign["reports"]["baseline_sync_json"])
PY
)

STATE_OUTPUT="${cfg[0]}"
REGISTRATION_OUTPUT="${cfg[1]}"
BASELINE_OUTPUT="${cfg[2]}"
BASELINE_CSV="${cfg[3]}"
BASELINE_JSON="${cfg[4]}"

python -m psmn_rl.analysis.lss_portfolio_campaign state-reconciliation \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$STATE_OUTPUT"

python -m psmn_rl.analysis.lss_portfolio_campaign registration \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REGISTRATION_OUTPUT"

python -m psmn_rl.analysis.lss_portfolio_campaign baseline-sync \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$BASELINE_OUTPUT" \
  --csv "$BASELINE_CSV" \
  --json "$BASELINE_JSON"
