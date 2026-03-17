# Benchmark Pack Schema Report

- frozen pack schema version: `1`
- candidate pack schema version: `1`

## Frozen Benchmark Pack Required Fields

- `schema_version`
- `pack_type`
- `claim`
- `canonical_method`
- `evaluation`
- `seed_groups`
- `variants`
- `thresholds`
- `thaw_gate`
- `candidate_pack`
- `manifest_reference`
- `authoritative_artifacts`
- `provenance`

## Frozen Benchmark Artifact Keys

- `frozen_claim_envelope`
- `manifest_report`
- `frozen_validation_report`
- `frozen_validation_csv`
- `frozen_validation_json`
- `claim_gate_dry_run`
- `claim_gate_dry_run_json`
- `claim_ledger`
- `future_retry_template`
- `freeze_hardening_decision_memo`
- `combined_doorkey_report`
- `combined_doorkey_csv`
- `final_block_report`
- `final_block_csv`
- `forensic_casebook`
- `forensic_round_audit`
- `forensic_route_locality`
- `forensic_decision_memo`
- `resume_scorecard`
- `keycorridor_transfer_report`
- `keycorridor_transfer_csv`

## Validator Checks

- manifest path exists and hash matches the sealed manifest reference
- pack schema version is recognized
- sealed claim, evaluation, variants, thresholds, and thaw gate still match the manifest
- every required authoritative artifact exists and still matches its sealed hash

A pack passes only if every required field, manifest check, and artifact hash check passes.
