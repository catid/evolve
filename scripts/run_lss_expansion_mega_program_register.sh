#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_expansion_mega_program/campaign.yaml}"

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

python -m psmn_rl.analysis.lss_expansion_mega_program state-reconciliation \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$STATE_OUTPUT"

python -m psmn_rl.analysis.lss_expansion_mega_program registration \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REGISTRATION_OUTPUT"

python -m psmn_rl.analysis.lss_expansion_mega_program baseline-sync \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$BASELINE_OUTPUT" \
  --csv "$BASELINE_CSV" \
  --json "$BASELINE_JSON"
