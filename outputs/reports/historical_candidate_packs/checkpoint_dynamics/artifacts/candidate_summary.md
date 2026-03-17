# Historical Candidate Pack: checkpoint_dynamics

- family: `PPO tuning / checkpoint dynamics`
- ledger family: `PPO tuning / checkpoint dynamics`
- historical status: `negative`
- expected verdict under current gate: `FAIL: claim remains frozen`
- rationale: PPO-only routed win language is now explicitly disallowed, and the archived checkpoint scan also misses the current external-64 fairness prerequisites.
- evaluation: `checkpoint_dynamics_scan` / `32` episodes
- requested claims: `['ppo_only_routed_win']`
- controls present: `['baseline_sare']`

## Source Reports

- `outputs/reports/checkpoint_dynamics_report.md`
- `outputs/reports/checkpoint_dynamics_report.csv`

## Provenance Note

The historical phase was a PPO-only checkpoint scan, not a teacher-guided KL learner-state comparison.
