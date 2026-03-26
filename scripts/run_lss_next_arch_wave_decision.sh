#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_next_arch_wave_program/campaign.yaml}"

source .venv/bin/activate
./.venv/bin/python -m psmn_rl.analysis.lss_next_arch_wave \
  --campaign-config "$CAMPAIGN_CONFIG" \
  write-candidate-pack

./.venv/bin/python -m psmn_rl.analysis.lss_next_arch_wave \
  --campaign-config "$CAMPAIGN_CONFIG" \
  write-decision-memo
