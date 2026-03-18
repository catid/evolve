#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_successor_migration/campaign.yaml}"

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
reports = campaign["reports"]
print(reports["registration"])
print(reports["baseline_sync"])
print(reports["baseline_csv"])
print(reports["baseline_json"])
PY
)

REGISTRATION_OUTPUT="${cfg[0]}"
BASELINE_OUTPUT="${cfg[1]}"
BASELINE_CSV="${cfg[2]}"
BASELINE_JSON="${cfg[3]}"

python -m psmn_rl.analysis.lss_successor_migration registration \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REGISTRATION_OUTPUT"

python -m psmn_rl.analysis.lss_successor_migration baseline-sync \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$BASELINE_OUTPUT" \
  --csv "$BASELINE_CSV" \
  --json "$BASELINE_JSON"
