# Candidate Result Pack Schema

- schema version: `1`
- frozen pack reference: `outputs/reports/frozen_benchmark_pack.json`

## Required Fields

- `schema_version`
- `pack_type`
- `candidate_name`
- `frozen_pack_reference`
- `task`
- `evaluation`
- `requested_claims`
- `controls_present`
- `metrics`
- `actual_sets`
- `artifacts`
- `provenance`

## Required Metrics

- both `combined` and `retry_block` must include every canonical variant
- each variant must expose: `mean, min, max, complete_seed_failures, seed_count`

## Required Artifact Roles

- `candidate_summary_markdown`
- `candidate_metrics_json`
- `combined_report_markdown`
- `combined_report_csv`
- `retry_block_report_markdown`
- `retry_block_report_csv`

## Validator Checks

- pack references the current sealed frozen benchmark pack path, claim id, and hash
- evaluation stays on external 64-episode DoorKey
- required fairness controls are present
- combined and retry-block lane/seed coverage matches the frozen benchmark
- required artifact roles exist and their hashes match the filesystem

Malformed or incomplete candidate packs are treated as `INCONCLUSIVE: missing prerequisites` by the pack-based claim gate.
