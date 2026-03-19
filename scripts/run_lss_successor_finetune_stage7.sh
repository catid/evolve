#!/usr/bin/env bash
set -euo pipefail

PSMN_CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_successor_finetune_league/campaign.yaml}" \
  bash ./scripts/run_lss_expansion_mega_program_stage7.sh
