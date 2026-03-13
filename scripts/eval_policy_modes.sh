#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH="${1:?checkpoint path required}"
CONFIG_PATH="${2:?config path required}"
EPISODES="${3:-16}"
DEVICE="${PSMN_DEVICE:-auto}"

echo "== greedy =="
python -m psmn_rl.evaluate \
  --checkpoint "$CHECKPOINT_PATH" \
  --config "$CONFIG_PATH" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --greedy true

echo
echo "== sampled =="
python -m psmn_rl.evaluate \
  --checkpoint "$CHECKPOINT_PATH" \
  --config "$CONFIG_PATH" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --greedy false
