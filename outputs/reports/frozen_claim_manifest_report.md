# Frozen Claim Manifest Report

- manifest: `configs/claims/doorkey_frozen_claim.yaml`
- canonical method: teacher `flat_dense` + `teacher_logit_kl` + `append_all`
- evaluation path: `external 64-episode policy_diagnostics path`

## Canonical Variants

| Variant Key | Display | Family |
| --- | --- | --- |
| `recovered_token_dense` | recovered token_dense | `baseline` |
| `kl_lss_token_dense` | KL learner-state token_dense | `teacher_guided_structured_control` |
| `kl_lss_single_expert` | KL learner-state single_expert | `teacher_guided_structured_control` |
| `baseline_sare` | baseline PPO SARE | `baseline` |
| `kl_lss_sare` | KL learner-state SARE | `teacher_guided_routed` |

## Canonical Seed Groups

- `structured_slice`: `[7, 11, 19, 23, 29, 31, 37, 41, 43]`
- `retry_block`: `[47, 53, 59]`
- `combined`: `[('original', [7, 11, 19]), ('fresh', [23, 29, 31]), ('fresh_extra', [37, 41, 43]), ('fresh_final', [47, 53, 59])]`

## Frozen Thresholds

- combined KL learner-state `SARE` mean: `0.7122`
- combined KL learner-state `single_expert` mean: `0.6862`
- retry-block KL learner-state `SARE` mean: `0.3125`
- retry-block KL learner-state `single_expert` mean: `0.4635`

## Pack Schema

- frozen benchmark pack schema version: `1`
- candidate result pack schema version: `1`
- required candidate artifact roles: `['candidate_summary_markdown', 'candidate_metrics_json', 'combined_report_markdown', 'combined_report_csv', 'retry_block_report_markdown', 'retry_block_report_csv']`

## Authoritative Reports

- `frozen_claim_envelope`: `outputs/reports/frozen_claim_envelope.md`
- `manifest_report`: `outputs/reports/frozen_claim_manifest_report.md`
- `frozen_validation_report`: `outputs/reports/frozen_baseline_validation.md`
- `frozen_validation_csv`: `outputs/reports/frozen_baseline_validation.csv`
- `frozen_validation_json`: `outputs/reports/frozen_baseline_validation.json`
- `claim_gate_dry_run`: `outputs/reports/claim_gate_dry_run.md`
- `claim_gate_dry_run_json`: `outputs/reports/claim_gate_dry_run.json`
- `claim_ledger`: `outputs/reports/claim_ledger.md`
- `future_retry_template`: `outputs/reports/future_retry_template.md`
- `freeze_hardening_decision_memo`: `outputs/reports/freeze_hardening_decision_memo.md`
- `combined_doorkey_report`: `outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.md`
- `combined_doorkey_csv`: `outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.csv`
- `final_block_report`: `outputs/reports/lss_final_block_single_expert_control_report.md`
- `final_block_csv`: `outputs/reports/lss_final_block_single_expert_control_report.csv`
- `forensic_casebook`: `outputs/reports/lss_forensic_casebook.md`
- `forensic_round_audit`: `outputs/reports/lss_forensic_round_audit.md`
- `forensic_route_locality`: `outputs/reports/lss_forensic_route_locality.md`
- `forensic_decision_memo`: `outputs/reports/lss_forensic_atlas_decision_memo.md`
- `resume_scorecard`: `outputs/reports/lss_resume_qualification_scorecard.md`
- `keycorridor_transfer_report`: `outputs/reports/lss_keycorridor_transfer_report.md`
- `keycorridor_transfer_csv`: `outputs/reports/lss_keycorridor_transfer_report.csv`
