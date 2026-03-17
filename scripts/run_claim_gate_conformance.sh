#!/usr/bin/env bash
set -euo pipefail

FROZEN_PACK=${1:-outputs/reports/frozen_benchmark_pack.json}
BASE_CANDIDATE=${2:-outputs/reports/frozen_candidate_result_pack.json}
CORPUS=${3:-tests/data/claim_gate_corpus/corpus.yaml}
GOLDEN=${4:-tests/data/claim_gate_corpus/golden_snapshot.json}
OUTPUT_DIR=${5:-outputs/reports/claim_gate_corpus}

python -m psmn_rl.analysis.claim_gate_conformance \
  --frozen-pack "$FROZEN_PACK" \
  --base-candidate "$BASE_CANDIDATE" \
  --corpus "$CORPUS" \
  --golden-snapshot "$GOLDEN" \
  --output-dir "$OUTPUT_DIR"
