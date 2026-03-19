#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_portfolio_campaign/campaign.yaml}"

PSMN_CAMPAIGN_CONFIG="$CAMPAIGN_CONFIG" bash ./scripts/run_lss_expansion_mega_program_stage4.sh
