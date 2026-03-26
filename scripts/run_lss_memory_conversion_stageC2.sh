#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

./.venv/bin/python -m psmn_rl.analysis.lss_memory_conversion stage-c2 --device auto
