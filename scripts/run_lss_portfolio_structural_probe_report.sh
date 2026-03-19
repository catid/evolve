#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_portfolio_structural_probe/campaign.yaml}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_portfolio_structural_probe \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output outputs/reports/portfolio_structural_probe.md \
  --json outputs/reports/portfolio_structural_probe.json
