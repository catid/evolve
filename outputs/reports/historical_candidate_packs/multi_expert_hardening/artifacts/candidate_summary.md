# Historical Candidate Pack: multi_expert_hardening

- family: `Multi-expert hardening`
- ledger family: `Multi-expert hardening`
- historical status: `bounded positive`
- expected verdict under current gate: `INCONCLUSIVE: missing prerequisites`
- rationale: The historical hardening phase still lacked the final same-block single_expert retry metrics needed by the current gate.
- evaluation: `external_policy_diagnostics` / `64` episodes
- requested claims: `['bounded_teacher_guided_doorkey_sare']`
- controls present: `['recovered_token_dense', 'kl_lss_token_dense', 'kl_lss_single_expert', 'baseline_sare', 'kl_lss_sare']`

## Source Reports

- `outputs/reports/lss_final_fresh_seed_block_report.md`
- `outputs/reports/lss_final_combined_doorkey_report.md`
- `outputs/reports/lss_multi_expert_hardening_decision_memo.md`

## Provenance Note

This pre-frozen phase is historically important because it first exposed the final-block reversal, but it did not yet include the later same-block single_expert closure.
