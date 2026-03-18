#!/usr/bin/env bash
set -euo pipefail

MANIFEST=${1:-configs/claims/doorkey_frozen_claim.yaml}
OUTPUT=${PSMN_FROZEN_VALIDATION_OUTPUT:-outputs/reports/frozen_baseline_validation.md}
CSV_OUTPUT=${PSMN_FROZEN_VALIDATION_CSV:-outputs/reports/frozen_baseline_validation.csv}
JSON_OUTPUT=${PSMN_FROZEN_VALIDATION_JSON:-outputs/reports/frozen_baseline_validation.json}

python -m psmn_rl.analysis.freeze_hardening validate-frozen-baseline \
  --manifest "$MANIFEST" \
  --output "$OUTPUT" \
  --csv "$CSV_OUTPUT" \
  --json-output "$JSON_OUTPUT"
