#!/usr/bin/env bash
set -euo pipefail

MANIFEST=${1:-configs/claims/doorkey_frozen_claim.yaml}
PACK_MD=${2:-outputs/reports/frozen_benchmark_pack.md}
PACK_JSON=${3:-outputs/reports/frozen_benchmark_pack.json}
VALIDATION_MD=${4:-outputs/reports/frozen_benchmark_pack_validation.md}
VALIDATION_JSON=${5:-outputs/reports/frozen_benchmark_pack_validation.json}
SOURCE_COMMIT=${FROZEN_PACK_SOURCE_COMMIT:-}
SOURCE_DIRTY=${FROZEN_PACK_SOURCE_DIRTY:-}

SEAL_ARGS=()
if [[ -n "$SOURCE_COMMIT" ]]; then
  SEAL_ARGS+=(--source-commit "$SOURCE_COMMIT")
fi
if [[ -n "$SOURCE_DIRTY" ]]; then
  SEAL_ARGS+=(--source-dirty "$SOURCE_DIRTY")
fi

python -m psmn_rl.analysis.benchmark_pack seal-frozen-pack \
  --manifest "$MANIFEST" \
  --output-markdown "$PACK_MD" \
  --output-json "$PACK_JSON" \
  "${SEAL_ARGS[@]}"

python -m psmn_rl.analysis.benchmark_pack validate-frozen-pack \
  --pack "$PACK_JSON" \
  --output "$VALIDATION_MD" \
  --json-output "$VALIDATION_JSON"
