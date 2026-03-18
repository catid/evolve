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
print(campaign["reports"]["stage6_report"])
print(campaign["reports"]["stage6_csv"])
print(campaign["reports"]["stage6_json"])
PY
)

REPORT_OUTPUT="${cfg[0]}"
REPORT_CSV="${cfg[1]}"
REPORT_JSON="${cfg[2]}"

python -m psmn_rl.analysis.lss_successor_migration stage6-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
