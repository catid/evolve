#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_portfolio_campaign/campaign.yaml}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_portfolio_campaign fairness-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output outputs/reports/portfolio_stage3_fairness.md \
  --csv outputs/reports/portfolio_stage3_fairness.csv \
  --json outputs/reports/portfolio_stage3_fairness.json
