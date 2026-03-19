# Portfolio Seed Pack

- source campaign: `lss_portfolio_campaign`
- git commit: `59cfc84c2b4e0ebd4f1a3ea9c5a85f4854836d62`
- git dirty: `True`
- active benchmark: `round6`

## Seed Roles

| Role | Lane | Seed | Requirement |
| --- | --- | ---: | --- |
| `sentinel` | `prospective_c` | 193 | `track_only` |
| `ranking_support` | `prospective_f` | 233 | `>= 1.0000` |
| `ranking_weakness` | `prospective_h` | 269 | `> 0.9844` |
| `guardrail` | `prospective_h` | 277 | `>= 1.0000` |

## Frontier Tiers

- restart default: `['round7']`
- reserve priors: `['round10', 'round5']`
- retired priors: `['door3_post5', 'post_unlock_x5']`

## Candidate Rows

| Candidate | Tier | Policy Bucket | Screen Rule | support_233 | weakness_269 | guardrail_277 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `round7` | `restart_default` | `restart_default` | `advance_for_broader_dev` | `1.0000` | `1.0000` | `1.0000` |
| `round10` | `reserve` | `reserve_same_signal_higher_cost` | `advance_for_broader_dev` | `1.0000` | `1.0000` | `1.0000` |
| `round5` | `reserve` | `reserve_below_incumbent` | `advance_for_broader_dev` | `1.0000` | `1.0000` | `1.0000` |
| `door3_post5` | `retired` | `retire_structural_regression` | `prune_support_regression` | `0.4531` | `1.0000` | `1.0000` |
| `post_unlock_x5` | `retired` | `retire_local_only_fix` | `prune_guardrail_regression` | `n/a` | `1.0000` | `0.8906` |

## Interpretation

- This pack is the operational frontier artifact for future bounded mini-sweeps.
- `round7` is the only restart-default line; `round10` and `round5` are reserve lines; `door3_post5` and `post_unlock_x5` are retired.
- A mini-sweep should consume these seed roles and screen rules before it spends broader DoorKey dev budget.
