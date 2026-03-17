#!/usr/bin/env bash
set -euo pipefail

MANIFEST=${1:-configs/claims/doorkey_frozen_claim.yaml}

python -m psmn_rl.analysis.freeze_hardening claim-envelope-report \
  --manifest "$MANIFEST" \
  --output outputs/reports/frozen_claim_envelope.md

python -m psmn_rl.analysis.freeze_hardening manifest-report \
  --manifest "$MANIFEST" \
  --output outputs/reports/frozen_claim_manifest_report.md

./scripts/run_frozen_baseline_validation.sh "$MANIFEST"

python -m psmn_rl.analysis.claim_gate \
  --manifest "$MANIFEST" \
  --candidate outputs/reports/frozen_baseline_validation.json \
  --output outputs/reports/claim_gate_dry_run.md \
  --json-output outputs/reports/claim_gate_dry_run.json

python -m psmn_rl.analysis.freeze_hardening claim-ledger \
  --output outputs/reports/claim_ledger.md

python -m psmn_rl.analysis.freeze_hardening future-retry-template \
  --manifest "$MANIFEST" \
  --output outputs/reports/future_retry_template.md

python -m psmn_rl.analysis.freeze_hardening decision-memo \
  --manifest "$MANIFEST" \
  --validation-json outputs/reports/frozen_baseline_validation.json \
  --claim-gate-json outputs/reports/claim_gate_dry_run.json \
  --output outputs/reports/freeze_hardening_decision_memo.md

./scripts/run_frozen_benchmark_pack_validation.sh \
  "$MANIFEST" \
  outputs/reports/frozen_benchmark_pack.md \
  outputs/reports/frozen_benchmark_pack.json \
  outputs/reports/frozen_benchmark_pack_validation.md \
  outputs/reports/frozen_benchmark_pack_validation.json

python -m psmn_rl.analysis.benchmark_pack benchmark-pack-schema-report \
  --manifest "$MANIFEST" \
  --output outputs/reports/benchmark_pack_schema_report.md

python -m psmn_rl.analysis.benchmark_pack candidate-pack-schema \
  --frozen-pack outputs/reports/frozen_benchmark_pack.json \
  --output-markdown outputs/reports/candidate_result_pack_schema.md \
  --output-json-template outputs/reports/candidate_result_pack_template.json

python -m psmn_rl.analysis.benchmark_pack build-current-frozen-candidate-pack \
  --manifest "$MANIFEST" \
  --frozen-pack outputs/reports/frozen_benchmark_pack.json \
  --candidate-json outputs/reports/frozen_baseline_validation.json \
  --output outputs/reports/frozen_candidate_result_pack.json

./scripts/run_claim_gate.sh \
  outputs/reports/frozen_benchmark_pack.json \
  outputs/reports/frozen_candidate_result_pack.json \
  outputs/reports/claim_gate_pack_dry_run.md \
  outputs/reports/claim_gate_pack_dry_run.json

python -m psmn_rl.analysis.benchmark_pack build-incomplete-candidate-pack \
  --frozen-pack outputs/reports/frozen_benchmark_pack.json \
  --output outputs/reports/candidate_result_pack_incomplete.json

./scripts/run_claim_gate.sh \
  outputs/reports/frozen_benchmark_pack.json \
  outputs/reports/candidate_result_pack_incomplete.json \
  outputs/reports/claim_gate_pack_inconclusive.md \
  outputs/reports/claim_gate_pack_inconclusive.json

python -m psmn_rl.analysis.freeze_hardening ci-claim-gate-report \
  --frozen-pack-validation-json outputs/reports/frozen_benchmark_pack_validation.json \
  --claim-gate-pack-json outputs/reports/claim_gate_pack_dry_run.json \
  --claim-gate-incomplete-json outputs/reports/claim_gate_pack_inconclusive.json \
  --output outputs/reports/ci_claim_gate_report.md

python -m psmn_rl.analysis.freeze_hardening operational-memo \
  --frozen-pack outputs/reports/frozen_benchmark_pack.json \
  --frozen-pack-validation-json outputs/reports/frozen_benchmark_pack_validation.json \
  --claim-gate-pack-json outputs/reports/claim_gate_pack_dry_run.json \
  --claim-gate-incomplete-json outputs/reports/claim_gate_pack_inconclusive.json \
  --output outputs/reports/freeze_hardening_operational_memo.md
