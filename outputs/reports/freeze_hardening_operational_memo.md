# Freeze-Hardening Operational Memo

## Answers

1. The frozen benchmark pack is `outputs/reports/frozen_benchmark_pack.json`, schema version `1`, sealing the claim `bounded teacher-guided DoorKey SARE result` against the canonical DoorKey artifacts and manifest.
2. The pack is validated by hash and schema. The current dry run returns `PASS: frozen benchmark pack validated`. See [frozen_benchmark_pack_validation.md](frozen_benchmark_pack_validation.md).
3. Any future candidate must be packaged as a candidate result pack with the required controls, metrics, actual lane/seed sets, and hashed artifacts. See [candidate_result_pack_schema.md](candidate_result_pack_schema.md) and [candidate_result_pack_template.json](candidate_result_pack_template.json).
4. The automated gate validates both packs first, then applies the frozen DoorKey thresholds: retry-block KL learner-state `SARE` must beat `0.3125`, match or beat same-block `single_expert`, and preserve combined KL learner-state `SARE` mean `0.7122`. The current dry run returns `FAIL: claim remains frozen` and malformed candidates return `INCONCLUSIVE: missing prerequisites`. See [claim_gate_pack_dry_run.md](claim_gate_pack_dry_run.md).
5. Repo workflow now includes a claim-gate GitHub Action and a PR template that require claim scope, candidate pack path, gate result, and retry-template reference for claim-sensitive changes. See [ci_claim_gate_report.md](ci_claim_gate_report.md).
6. Yes. The project is now operationally frozen until a preregistered retry clears the pack-based gate.
