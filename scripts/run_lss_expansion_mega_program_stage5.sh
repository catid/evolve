#!/usr/bin/env bash
set -euo pipefail

PSMN_CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_expansion_mega_program/campaign.yaml}" \
  bash ./scripts/run_lss_successor_migration_stage4.sh
