# Historical Candidate Pack: teacher_extraction_multiseed

- family: `Learner-state supervision`
- ledger family: `Learner-state supervision`
- historical status: `bounded positive`
- expected verdict under current gate: `INCONCLUSIVE: missing prerequisites`
- rationale: The reopened positive signal lacked the modern token_dense KL control, single_expert control, and canonical retry block.
- evaluation: `external_policy_diagnostics` / `64` episodes
- requested claims: `['bounded_teacher_guided_doorkey_sare']`
- controls present: `['recovered_token_dense', 'baseline_sare', 'kl_lss_sare']`

## Source Reports

- `outputs/reports/teacher_extraction_multiseed_report.md`
- `outputs/reports/teacher_extraction_multiseed_report.csv`

## Provenance Note

The original 3-seed reopened result predated the later matched-control and final-block fairness requirements.
