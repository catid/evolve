#!/usr/bin/env bash
set -euo pipefail

export PSMN_CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_portfolio_structural_probe/campaign.yaml}"
bash ./scripts/run_lss_successor_migration_stage1.sh
