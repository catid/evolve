# Portfolio Seed Pack Validation

- source campaign: `lss_portfolio_campaign`
- git commit: `b430b8f7454f1ed11be1686b488c6a44574fada8`
- git dirty: `True`
- active benchmark: `round6`

## Historical Consistency Check

| Candidate | Tier | Screen Rule | Track | Family | Dev Mean | Delta vs round6 | Stage1 Reason | Validation |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- |
| `round7` | `restart_default` | `advance_for_broader_dev` | `fruitful` | `near_neighbor_rounds` | `0.8889` | `0.0000` | `stop: outside fruitful top-3` | `validated_restart_default` |
| `round10` | `reserve` | `advance_for_broader_dev` | `fruitful` | `near_neighbor_rounds` | `0.8889` | `0.0000` | `advance` | `validated_reserve` |
| `round5` | `reserve` | `advance_for_broader_dev` | `fruitful` | `near_neighbor_rounds` | `0.8368` | `-0.0521` | `stop: below incumbent dev mean` | `validated_reserve` |
| `door3_post5` | `retired` | `prune_support_regression` | `pack_only` | `pack_only` | `n/a` | `n/a` | `pack_only` | `validated_retired` |
| `post_unlock_x5` | `retired` | `prune_guardrail_regression` | `pack_only` | `pack_only` | `n/a` | `n/a` | `pack_only` | `validated_retired` |

## Validation Summary

- validated restart default: `['round7']`
- validated reserve priors: `['round10', 'round5']`
- validated retired priors: `['door3_post5', 'post_unlock_x5']`
- needs review: `[]`

## Interpretation

- This validator checks that the machine-readable seed pack agrees with the broader portfolio Stage 1 evidence rather than only restating the packed seed values.
- `round7` should validate as the only restart default because it kept the required seed behavior while staying at incumbent dev mean without the extra cost of `round10` or the broader dev drop of `round5`.
- `door3_post5` and `post_unlock_x5` should validate as retired because the packed prune rules match their measured support or guardrail regressions.
