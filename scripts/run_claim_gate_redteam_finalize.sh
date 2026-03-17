#!/usr/bin/env bash
set -euo pipefail

MANIFEST=${1:-configs/claims/doorkey_frozen_claim.yaml}

./scripts/run_freeze_hardening_finalize.sh "$MANIFEST"

./scripts/run_frozen_benchmark_pack_validation.sh \
  "$MANIFEST" \
  outputs/reports/frozen_benchmark_pack.md \
  outputs/reports/frozen_benchmark_pack.json \
  outputs/reports/frozen_benchmark_pack_validation.md \
  outputs/reports/frozen_benchmark_pack_validation.json

./scripts/run_claim_gate.sh \
  outputs/reports/frozen_benchmark_pack.json \
  outputs/reports/frozen_candidate_result_pack.json \
  outputs/reports/claim_gate_pack_dry_run.md \
  outputs/reports/claim_gate_pack_dry_run.json

bash ./scripts/run_claim_gate_conformance.sh \
  outputs/reports/frozen_benchmark_pack.json \
  outputs/reports/frozen_candidate_result_pack.json \
  tests/data/claim_gate_corpus/corpus.yaml \
  tests/data/claim_gate_corpus/golden_snapshot.json \
  outputs/reports/claim_gate_corpus
