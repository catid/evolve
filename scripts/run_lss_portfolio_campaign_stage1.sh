#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_portfolio_campaign/campaign.yaml}"

PSMN_CAMPAIGN_CONFIG="$CAMPAIGN_CONFIG" bash ./scripts/run_lss_expansion_mega_program_stage1.sh

source .venv/bin/activate
python -m psmn_rl.analysis.lss_portfolio_campaign stage1-screening \
  --campaign-config "$CAMPAIGN_CONFIG"
