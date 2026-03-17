#!/usr/bin/env bash
set -euo pipefail

FROZEN_PACK=${1:-outputs/reports/frozen_benchmark_pack.json}
CATALOG=${2:-tests/data/claim_history_replay/catalog.yaml}
GOLDEN=${3:-tests/data/claim_history_replay/golden_snapshot.json}
OUTPUT_DIR=${4:-outputs/reports/historical_candidate_packs}

python -m psmn_rl.analysis.claim_history_replay \
  --frozen-pack "$FROZEN_PACK" \
  --catalog "$CATALOG" \
  --golden-snapshot "$GOLDEN" \
  --output-dir "$OUTPUT_DIR"
