# CI Claim Gate Report

## Workflow

- GitHub Actions workflow: `.github/workflows/claim-gate.yml`
- Pull request template: `.github/pull_request_template.md`

## Local Entry Points

- `./scripts/run_frozen_benchmark_pack_validation.sh`
- `./scripts/run_claim_gate.sh`
- `./scripts/run_freeze_hardening_finalize.sh`

## Current Dry Runs

- frozen benchmark pack validation: `PASS: frozen benchmark pack validated`
- pack-based claim gate on the current frozen candidate: `FAIL: claim remains frozen`
- malformed/incomplete candidate pack gate: `INCONCLUSIVE: missing prerequisites`

## Enforcement Rule

- Any future thaw candidate should include a candidate result pack path, a pack-based gate report, and a retry-template reference before claim language is discussed in review.
