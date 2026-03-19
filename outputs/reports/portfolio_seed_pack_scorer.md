# Portfolio Seed Pack Scorer

- source campaign: `lss_portfolio_campaign`
- git commit: `94f889602abc67b3a0b5b0c924090b4f17b09a1c`
- git dirty: `True`
- active benchmark: `round6`

## Candidate Verdicts

| Candidate | Tier | support_233 | weakness_269 | guardrail_277 | Dev Mean | Delta vs round6 | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `round7` | `restart_default` | `1.0000` | `1.0000` | `1.0000` | `0.8889` | `0.0000` | `advance_for_broader_dev` |
| `round10` | `reserve` | `1.0000` | `1.0000` | `1.0000` | `0.8889` | `0.0000` | `advance_for_broader_dev` |
| `round5` | `reserve` | `1.0000` | `1.0000` | `1.0000` | `0.8368` | `-0.0521` | `hold_seed_clean_but_below_incumbent` |
| `door3_post5` | `retired` | `0.4531` | `1.0000` | `1.0000` | `n/a` | `n/a` | `prune_support_regression` |
| `post_unlock_x5` | `retired` | `n/a` | `1.0000` | `0.8906` | `n/a` | `n/a` | `prune_guardrail_regression` |

## Verdict Groups

- groups: `{'advance_for_broader_dev': ['round7', 'round10'], 'hold_seed_clean_but_below_incumbent': ['round5'], 'prune_guardrail_regression': ['post_unlock_x5'], 'prune_support_regression': ['door3_post5']}`

## Interpretation

- This scorer is the executable pre-fairness gate for future bounded mini-sweeps built on the portfolio seed pack.
- It prunes support, guardrail, and weakness regressions immediately; it holds exact weakness ties; and it only advances seed-clean lines that also avoid a broader dev drop when that dev summary is available.
- On the current measured frontier, `round7` and `round10` should advance for broader dev, `round5` should be held as seed-clean but below-incumbent, and the retired lines should be pruned.
