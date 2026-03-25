#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_next_wave_program/campaign.yaml}"

source .venv/bin/activate
./.venv/bin/python -m psmn_rl.analysis.lss_next_wave \
  --campaign-config "$CAMPAIGN_CONFIG" \
  write-recurrent-impl-report

./.venv/bin/python -m psmn_rl.analysis.lss_next_wave \
  --campaign-config "$CAMPAIGN_CONFIG" \
  run-recurrent-controls \
  --device "$DEVICE"

./.venv/bin/python -m psmn_rl.analysis.lss_next_wave \
  --campaign-config "$CAMPAIGN_CONFIG" \
  write-recurrent-controls-report

./.venv/bin/python -m psmn_rl.analysis.lss_next_wave \
  --campaign-config "$CAMPAIGN_CONFIG" \
  run-screen \
  --family por \
  --device "$DEVICE"

./.venv/bin/python -m psmn_rl.analysis.lss_next_wave \
  --campaign-config "$CAMPAIGN_CONFIG" \
  write-family-report \
  --family por
