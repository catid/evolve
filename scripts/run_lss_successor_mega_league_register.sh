#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_successor_mega_league/campaign.yaml}"

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
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

python -m psmn_rl.analysis.lss_successor_mega_league state-reconciliation \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$STATE_OUTPUT"

python -m psmn_rl.analysis.lss_successor_mega_league registration \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REGISTRATION_OUTPUT"

python -m psmn_rl.analysis.lss_successor_mega_league baseline-sync \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$BASELINE_OUTPUT" \
  --csv "$BASELINE_CSV" \
  --json "$BASELINE_JSON"
