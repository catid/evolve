# Historical Candidate Pack: resume_gate

- family: `Resume gate`
- ledger family: `Resume gate`
- historical status: `frozen`
- expected verdict under current gate: `FAIL: claim remains frozen`
- rationale: Once the final-block single_expert control exists, the current frozen candidate still fails the thaw bars.
- evaluation: `external_policy_diagnostics` / `64` episodes
- requested claims: `['bounded_teacher_guided_doorkey_sare']`
- controls present: `['recovered_token_dense', 'kl_lss_token_dense', 'kl_lss_single_expert', 'baseline_sare', 'kl_lss_sare']`

## Source Reports

- `outputs/reports/lss_final_block_single_expert_control_report.md`
- `outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.md`
- `outputs/reports/lss_resume_gate_decision_memo.md`

## Provenance Note

This is the first frozen-era phase where the modern fairness closure exists and the gate can evaluate the retry block directly.
