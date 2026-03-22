#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_mechanism_program/campaign.yaml}"

PSMN_CAMPAIGN_CONFIG="$CAMPAIGN_CONFIG" bash ./scripts/run_lss_expansion_mega_program_stage2.sh

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["stage2_verification_report"])
print(campaign["reports"]["stage2_verification_csv"])
print(campaign["reports"]["stage2_verification_json"])
PY
)

REPORT_OUTPUT="${cfg[0]}"
REPORT_CSV="${cfg[1]}"
REPORT_JSON="${cfg[2]}"

python -m psmn_rl.analysis.lss_portfolio_campaign verification-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
