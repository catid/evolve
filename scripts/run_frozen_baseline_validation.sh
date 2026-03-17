#!/usr/bin/env bash
set -euo pipefail

MANIFEST=${1:-configs/claims/doorkey_frozen_claim.yaml}

python -m psmn_rl.analysis.freeze_hardening validate-frozen-baseline \
  --manifest "$MANIFEST" \
  --output outputs/reports/frozen_baseline_validation.md \
  --csv outputs/reports/frozen_baseline_validation.csv \
  --json-output outputs/reports/frozen_baseline_validation.json
