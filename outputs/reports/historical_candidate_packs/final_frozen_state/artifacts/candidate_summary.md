# Historical Candidate Pack: final_frozen_state

- family: `Final frozen state`
- ledger family: `Final frozen state`
- historical status: `frozen`
- expected verdict under current gate: `FAIL: claim remains frozen`
- rationale: The current frozen candidate is the explicit no-thaw reference point and must still fail its own thaw gate.
- evaluation: `external_policy_diagnostics` / `64` episodes
- requested claims: `['bounded_teacher_guided_doorkey_sare']`
- controls present: `['recovered_token_dense', 'kl_lss_token_dense', 'kl_lss_single_expert', 'baseline_sare', 'kl_lss_sare']`

## Source Reports

- `outputs/reports/frozen_benchmark_pack.md`
- `outputs/reports/frozen_claim_envelope.md`
- `outputs/reports/freeze_hardening_operational_memo.md`
- `outputs/reports/lss_final_block_single_expert_control_report.md`
- `outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.md`

## Provenance Note

This is the current sealed frozen benchmark candidate and should remain the hard-fail thaw reference.
