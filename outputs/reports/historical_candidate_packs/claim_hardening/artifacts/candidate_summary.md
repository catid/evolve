# Historical Candidate Pack: claim_hardening

- family: `Claim hardening`
- ledger family: `Claim hardening`
- historical status: `bounded positive`
- expected verdict under current gate: `INCONCLUSIVE: missing prerequisites`
- rationale: The claim-hardening phase strengthened the narrow DoorKey signal but still lacked the final retry block and same-block single_expert fairness control.
- evaluation: `external_policy_diagnostics` / `64` episodes
- requested claims: `['bounded_teacher_guided_doorkey_sare']`
- controls present: `['recovered_token_dense', 'kl_lss_token_dense', 'baseline_sare', 'kl_lss_sare']`

## Source Reports

- `outputs/reports/lss_additional_seed_report.md`
- `outputs/reports/lss_matched_control_report.md`
- `outputs/reports/lss_claim_hardening_decision_memo.md`

## Provenance Note

This phase added matched token_dense controls and fresh seeds, but not the current single_expert and final retry-block closure.
