# Claim Gate Corpus Report

- corpus: `doorkey_frozen_claim_gate_redteam_corpus`
- case count: `21`

| Case | Category | Expected | Rationale | Candidate Pack |
| --- | --- | --- | --- | --- |
| `happy_path_reference_fail` | `happy_path_reference` | `FAIL: claim remains frozen` | Comparable DoorKey candidate stays inside scope but fails the retry-block and same-block single_expert bars. | `outputs/reports/claim_gate_corpus/packs/happy_path_reference_fail.json` |
| `synthetic_pass_reference` | `synthetic_pass` | `PASS: thaw consideration allowed` | Keeps the PASS path covered so the gate is not only tested on failures. | `outputs/reports/claim_gate_corpus/packs/synthetic_pass_reference.json` |
| `incomplete_missing_controls` | `incomplete` | `INCONCLUSIVE: missing prerequisites` | Missing fairness controls means the gate cannot compare the candidate cleanly. | `outputs/reports/claim_gate_corpus/packs/incomplete_missing_controls.json` |
| `incomplete_missing_retry_metrics` | `incomplete` | `INCONCLUSIVE: missing prerequisites` | Retry-block metrics are required to evaluate thaw eligibility. | `outputs/reports/claim_gate_corpus/packs/incomplete_missing_retry_metrics.json` |
| `incomplete_missing_combined_metrics` | `incomplete` | `INCONCLUSIVE: missing prerequisites` | Combined DoorKey preservation is a required thaw bar. | `outputs/reports/claim_gate_corpus/packs/incomplete_missing_combined_metrics.json` |
| `incomplete_missing_provenance_commit` | `incomplete` | `INCONCLUSIVE: missing prerequisites` | Candidate provenance must be complete and well formed before the pack is comparable. | `outputs/reports/claim_gate_corpus/packs/incomplete_missing_provenance_commit.json` |
| `schema_wrong_schema_version` | `schema_invalid` | `INCONCLUSIVE: missing prerequisites` | Unknown schema versions are not comparable to the sealed frozen pack. | `outputs/reports/claim_gate_corpus/packs/schema_wrong_schema_version.json` |
| `schema_wrong_controls_type` | `schema_invalid` | `INCONCLUSIVE: missing prerequisites` | controls_present must stay machine-readable for fairness validation. | `outputs/reports/claim_gate_corpus/packs/schema_wrong_controls_type.json` |
| `schema_missing_actual_sets` | `schema_invalid` | `INCONCLUSIVE: missing prerequisites` | Lane/seed coverage must be explicit so the gate cannot compare mismatched slices. | `outputs/reports/claim_gate_corpus/packs/schema_missing_actual_sets.json` |
| `schema_malformed_json` | `schema_invalid` | `INCONCLUSIVE: missing prerequisites` | Malformed structured input must surface as a concrete load failure rather than crashing the gate. | `outputs/reports/claim_gate_corpus/packs/schema_malformed_json.json` |
| `semantic_wrong_task` | `semantically_invalid` | `INCONCLUSIVE: missing prerequisites` | DoorKey-only thaw consideration cannot use a KeyCorridor candidate pack. | `outputs/reports/claim_gate_corpus/packs/semantic_wrong_task.json` |
| `semantic_wrong_eval_path` | `semantically_invalid` | `INCONCLUSIVE: missing prerequisites` | The frozen claim is defined on external 64-episode DoorKey evaluation only. | `outputs/reports/claim_gate_corpus/packs/semantic_wrong_eval_path.json` |
| `semantic_wrong_retry_seeds` | `semantically_invalid` | `INCONCLUSIVE: missing prerequisites` | Retry-block comparisons must stay on the canonical 47/53/59 slice. | `outputs/reports/claim_gate_corpus/packs/semantic_wrong_retry_seeds.json` |
| `near_miss_below_single_expert` | `near_miss` | `FAIL: claim remains frozen` | Beating the frozen SARE baseline alone is insufficient for thaw consideration. | `outputs/reports/claim_gate_corpus/packs/near_miss_below_single_expert.json` |
| `near_miss_regress_combined` | `near_miss` | `FAIL: claim remains frozen` | Thaw consideration requires preserving the full combined DoorKey picture. | `outputs/reports/claim_gate_corpus/packs/near_miss_regress_combined.json` |
| `near_miss_extra_seed_failure` | `near_miss` | `FAIL: claim remains frozen` | The retry block still fails if SARE picks up extra complete-seed failures. | `outputs/reports/claim_gate_corpus/packs/near_miss_extra_seed_failure.json` |
| `provenance_hash_mismatch` | `provenance_tampering` | `INCONCLUSIVE: missing prerequisites` | Artifact hash mismatches should invalidate the pack before any thaw comparison happens. | `outputs/reports/claim_gate_corpus/packs/provenance_hash_mismatch.json` |
| `provenance_missing_file` | `provenance_tampering` | `INCONCLUSIVE: missing prerequisites` | Missing support artifacts invalidate the candidate pack. | `outputs/reports/claim_gate_corpus/packs/provenance_missing_file.json` |
| `tampered_metrics_artifact_mismatch` | `provenance_tampering` | `INCONCLUSIVE: missing prerequisites` | Metrics/artifact mismatches are a direct candidate-pack tampering case and must be blocked. | `outputs/reports/claim_gate_corpus/packs/tampered_metrics_artifact_mismatch.json` |
| `overclaim_keycorridor_transfer` | `overclaim` | `FAIL: claim remains frozen` | Disallowed claim widening must stay a hard fail. | `outputs/reports/claim_gate_corpus/packs/overclaim_keycorridor_transfer.json` |
| `overclaim_missing_controls` | `overclaim` | `FAIL: claim remains frozen` | Missing controls must not downgrade an explicit overclaim into inconclusive status. | `outputs/reports/claim_gate_corpus/packs/overclaim_missing_controls.json` |
