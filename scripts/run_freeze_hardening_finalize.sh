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
./scripts/run_claim_gate.sh \
  "$MANIFEST" \
  outputs/reports/frozen_baseline_validation.json \
  outputs/reports/claim_gate_dry_run.md \
  outputs/reports/claim_gate_dry_run.json

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
