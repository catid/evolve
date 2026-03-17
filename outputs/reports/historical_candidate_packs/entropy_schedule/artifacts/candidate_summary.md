# Historical Candidate Pack: entropy_schedule

- family: `Entropy schedules`
- ledger family: `Entropy schedules`
- historical status: `negative`
- expected verdict under current gate: `FAIL: claim remains frozen`
- rationale: PPO-only routed win language is disallowed, and the entropy sweep also predates the canonical external-64 thaw prerequisites.
- evaluation: `entropy_schedule_sweep` / `32` episodes
- requested claims: `['ppo_only_routed_win']`
- controls present: `['baseline_sare']`

## Source Reports

- `outputs/reports/entropy_schedule_report.md`
- `outputs/reports/entropy_schedule_report.csv`

## Provenance Note

This is a PPO-only schedule sweep rather than a teacher-guided structured comparison.
