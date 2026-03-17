# Historical Candidate Pack: claim_broadening

- family: `Claim broadening`
- ledger family: `Claim broadening`
- historical status: `bounded positive`
- expected verdict under current gate: `FAIL: claim remains frozen`
- rationale: The historical broadening phase requested now-disallowed specifically multi-expert routed claim language.
- evaluation: `external_policy_diagnostics` / `64` episodes
- requested claims: `['bounded_teacher_guided_doorkey_sare', 'specifically_multi_expert_routed_advantage']`
- controls present: `['recovered_token_dense', 'kl_lss_token_dense', 'kl_lss_single_expert', 'baseline_sare', 'kl_lss_sare']`

## Source Reports

- `outputs/reports/lss_single_expert_matched_control_report.md`
- `outputs/reports/lss_extended_route_dependence_report.md`
- `outputs/reports/lss_additional_fresh_seed_block_report.md`
- `outputs/reports/lss_claim_broadening_decision_memo.md`

## Provenance Note

This pack intentionally reflects the stronger multi-expert wording that the current frozen claim envelope now disallows.
