#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH="${1:?checkpoint path required}"
CONFIG_PATH="${2:?config path required}"

python -m psmn_rl.evaluate --checkpoint "$CHECKPOINT_PATH" --config "$CONFIG_PATH"
