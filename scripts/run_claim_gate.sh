#!/usr/bin/env bash
set -euo pipefail

FROZEN_PACK=${1:-outputs/reports/frozen_benchmark_pack.json}
CANDIDATE_PACK=${2:-outputs/reports/frozen_candidate_result_pack.json}
OUTPUT=${3:-outputs/reports/claim_gate_pack_dry_run.md}
JSON_OUTPUT=${4:-outputs/reports/claim_gate_pack_dry_run.json}

python -m psmn_rl.analysis.claim_gate \
  --frozen-pack "$FROZEN_PACK" \
  --candidate-pack "$CANDIDATE_PACK" \
  --output "$OUTPUT" \
  --json-output "$JSON_OUTPUT"
