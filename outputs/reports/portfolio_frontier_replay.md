# Portfolio Frontier Replay

- source campaign: `lss_portfolio_campaign`
- git commit: `ddb5b00943b9d1e3a34e5e17209b396a65328f06`
- git dirty: `True`
- replay surface: candidates with measured `prospective_f/233` support plus historical `prospective_h/269/277` weakness coverage

## Measured Quartet Replay

| Candidate | Weakness Alias | Track | Family | c/193 | f/233 | h/269 | h/277 | Dev Mean | Delta vs round6 | Verdict |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `round10` | `round10` | `fruitful` | `near_neighbor_rounds` | `0.0000` | `1.0000` | `1.0000` | `1.0000` | `0.8889` | `0.0000` | `advance_for_broader_dev` |
| `round10_carry2_post4` | `carry2_post4` | `exploratory` | `bridge_weighting` | `0.0000` | `1.0000` | `1.0000` | `0.5000` | `0.8889` | `0.0000` | `prune_guardrail_regression` |
| `round10_door2_post4` | `door2_post4` | `exploratory` | `bridge_weighting` | `0.0000` | `1.0000` | `1.0000` | `0.6094` | `0.8889` | `0.0000` | `prune_guardrail_regression` |
| `round10_post_unlock_x5` | `post_unlock_x5` | `fruitful` | `hard_postunlock_weighting` | `0.0000` | `1.0000` | `1.0000` | `0.8906` | `0.8889` | `0.0000` | `prune_guardrail_regression` |

## Replay Summary

- verdict groups: `{'advance_for_broader_dev': ['round10'], 'prune_guardrail_regression': ['round10_carry2_post4', 'round10_door2_post4', 'round10_post_unlock_x5']}`

## Interpretation

- This replay scores only the historically measured quartet, not every historical candidate name, because most lines never received the full support-plus-weakness measurement surface.
- `round10` is the only fully measured historical line that remains seed-clean enough to deserve broader dev spending if the project reopens bounded search from the near-neighbor side.
- `round10_carry2_post4`, `round10_door2_post4`, and `round10_post_unlock_x5` all fail the `prospective_h/277` guardrail once the historical weakness surface is stitched in, so they should stay pruned despite looking clean on the narrower portfolio dev split alone.
- `prospective_c/193` stays all-zero across the replay surface, so it remains a sentinel rather than a ranking seed.
