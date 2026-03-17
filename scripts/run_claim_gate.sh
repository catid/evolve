#!/usr/bin/env bash
set -euo pipefail

MANIFEST=${1:-configs/claims/doorkey_frozen_claim.yaml}
CANDIDATE=${2:-outputs/reports/frozen_baseline_validation.json}
OUTPUT=${3:-outputs/reports/claim_gate_dry_run.md}
JSON_OUTPUT=${4:-outputs/reports/claim_gate_dry_run.json}

python -m psmn_rl.analysis.claim_gate \
  --manifest "$MANIFEST" \
  --candidate "$CANDIDATE" \
  --output "$OUTPUT" \
  --json-output "$JSON_OUTPUT"
