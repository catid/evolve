#!/usr/bin/env bash
set -euo pipefail

PSMN_CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_successor_mega_league/campaign.yaml}" \
  bash ./scripts/run_lss_successor_migration_stage3.sh
