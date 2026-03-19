# Portfolio Frontier Consistency Check

- source campaign: `lss_portfolio_campaign`
- git commit: `39e58497ed4e14d11c9669163e3f2e336353a40e`
- git dirty: `True`
- overall: `pass`

## Checks

| Check | Status | Detail |
| --- | --- | --- |
| `active_benchmark_round6` | `pass` | manifest active=['round6'] |
| `restart_default_round7` | `pass` | restart=['round7'] manifest=['round7'] pack=['round7'] |
| `retired_priors_match` | `pass` | restart=['door3_post5', 'post_unlock_x5'] manifest=['door3_post5', 'post_unlock_x5'] pack=['door3_post5', 'post_unlock_x5'] |
| `round7_scorer_advances` | `pass` | scorer round7=advance_for_broader_dev |
| `round10_replay_advances` | `pass` | replay round10=advance_for_broader_dev manifest=['round10'] |
| `round5_hold_only` | `pass` | scorer round5=hold_seed_clean_but_below_incumbent manifest=['round5'] |
| `door3_retired_consistent` | `pass` | scorer door3=prune_support_regression manifest=retired_prior |
| `post_unlock_x5_retired_consistent` | `pass` | scorer post_unlock_x5=prune_guardrail_regression manifest=retired_prior |

## Interpretation

- This is the drift check for the measured frontier stack.
- It should stay green as long as the restart policy, seed pack, scorer, replay, and manifest all encode the same operational state.
- Any future bounded search update that changes one frontier artifact without the others should trip this report before the repo starts giving contradictory guidance.
