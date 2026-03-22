# Portfolio Stage 8 Exploratory Transfer

- exploratory task: `KeyCorridor`
- evaluated lines: `['round6']`
- overall boundary classification: `clearly negative`
- historical recovered token_dense mean: `0.0000`
- historical baseline PPO SARE mean: `0.0000`

| Line | KL learner-state SARE | KL learner-state token_dense | KL learner-state single_expert | SARE-token | SARE-single | Failures | Boundary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `round6` | `0.0000` | `0.0000` | `n/a` | `0.0000` | `n/a` | `3` | `clearly negative` |

| Line | Seed | Variant | Greedy Success |
| --- | --- | --- | ---: |
| `round6` | 7 | KL learner-state SARE | 0.0000 |
| `round6` | 7 | KL learner-state token_dense | 0.0000 |
| `round6` | 11 | KL learner-state SARE | 0.0000 |
| `round6` | 11 | KL learner-state token_dense | 0.0000 |
| `round6` | 19 | KL learner-state SARE | 0.0000 |
| `round6` | 19 | KL learner-state token_dense | 0.0000 |

## Historical Reference

- `outputs/reports/lss_keycorridor_transfer_report.md` remains the historical bounded negative reference for KeyCorridor and is used here only as exploratory context, not as claim-widening evidence.
