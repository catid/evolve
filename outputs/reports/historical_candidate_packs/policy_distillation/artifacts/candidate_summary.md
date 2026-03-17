# Historical Candidate Pack: policy_distillation

- family: `Offline teacher distillation`
- ledger family: `Offline teacher distillation`
- historical status: `negative`
- expected verdict under current gate: `INCONCLUSIVE: missing prerequisites`
- rationale: The phase is teacher-guided but not the current KL learner-state family and does not supply the modern matched-control/retry-block comparison pack.
- evaluation: `offline_policy_distillation` / `64` episodes
- requested claims: `['bounded_teacher_guided_doorkey_sare']`
- controls present: `[]`

## Source Reports

- `outputs/reports/policy_distillation_report.md`
- `outputs/reports/policy_distillation_report.csv`

## Provenance Note

Historical distillation results are preserved as source evidence, but they are not directly comparable to the frozen KL learner-state benchmark slice.
