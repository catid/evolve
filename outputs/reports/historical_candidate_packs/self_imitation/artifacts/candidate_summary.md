# Historical Candidate Pack: self_imitation

- family: `Self-imitation`
- ledger family: `Self-imitation`
- historical status: `negative`
- expected verdict under current gate: `INCONCLUSIVE: missing prerequisites`
- rationale: The historical self-imitation phase did not package the required external-64 structured controls or retry-block slice under the modern gate.
- evaluation: `self_imitation_finetune` / `64` episodes
- requested claims: `['historical_recovery_probe']`
- controls present: `['baseline_sare']`

## Source Reports

- `outputs/reports/self_imitation_report.md`
- `outputs/reports/self_imitation_report.csv`

## Provenance Note

The report summarizes before/after greedy and sampled behavior, not a canonical external-64 structured comparison pack.
