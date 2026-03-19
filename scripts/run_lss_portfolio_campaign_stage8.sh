#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_portfolio_campaign/campaign.yaml}"

PSMN_CAMPAIGN_CONFIG="$CAMPAIGN_CONFIG" bash ./scripts/run_lss_expansion_mega_program_stage8.sh

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print(campaign["reports"]["stage8_report"])
print(campaign["reports"]["stage8_csv"])
print(campaign["reports"]["stage8_json"])
PY
)

REPORT_OUTPUT="${cfg[0]}"
REPORT_CSV="${cfg[1]}"
REPORT_JSON="${cfg[2]}"

python -m psmn_rl.analysis.lss_portfolio_campaign exploratory-transfer \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
