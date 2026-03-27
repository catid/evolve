#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

./.venv/bin/python -m psmn_rl.analysis.lss_memory_next --campaign-config configs/experiments/lss_memory_next_program/campaign.yaml stage-d3 --device auto
