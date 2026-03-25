#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_architecture_specialist_program/campaign.yaml}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_architecture_specialist stage-b4 \
  --campaign-config "$CAMPAIGN_CONFIG"
