# Claim Gate Hardening Report

This report records the concrete weaknesses exercised by the adversarial corpus and the hardening added to block them.

| Weakness | Fix | Covered Cases | Coverage |
| --- | --- | --- | --- |
| Malformed pack files could raise loader errors instead of producing a structured gate verdict. | The pack-based gate now uses safe structured loads and returns `INCONCLUSIVE: missing prerequisites` with concrete `*_pack_load` reasons. | `schema_malformed_json` | `PASS` |
| Disallowed claim widening could be downgraded to `INCONCLUSIVE` when the same pack also omitted controls or metrics. | Claim-scope failures now dominate pack-validation incompleteness, so overclaim attempts stay hard-failed. | `overclaim_keycorridor_transfer`, `overclaim_missing_controls` | `PASS` |
| Candidate-pack type, metrics, actual-set, artifact, and provenance fields accepted overly loose shapes. | The validator now enforces list/mapping/number/boolean field shapes and rejects malformed lane-seed payloads, duplicate artifact roles, and invalid git provenance fields. | `schema_wrong_schema_version`, `schema_wrong_controls_type`, `schema_missing_actual_sets`, `semantic_wrong_retry_seeds`, `provenance_missing_file` | `PASS` |
| A candidate pack could declare metrics that diverged from its `candidate_metrics_json` artifact. | The validator now parses `candidate_metrics_json` and checks that task, evaluation, claims, controls, metrics, and actual sets match the candidate pack. | `tampered_metrics_artifact_mismatch` | `PASS` |

## Outcome

- Malformed candidate files are now converted into structured `INCONCLUSIVE` gate results.
- Disallowed claim widening stays `FAIL` even when the same candidate also omits controls or metrics.
- Candidate packs now fail validation on malformed field shapes, bad provenance, and metrics-artifact mismatches.
