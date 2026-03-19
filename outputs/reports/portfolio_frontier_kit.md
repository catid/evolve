# Portfolio Frontier Kit

- source campaign: `lss_portfolio_campaign`
- git commit: `50c348e307a3d9baa27f3fcccebbb9feeb75c3a1`
- git dirty: `True`
- active benchmark: `round6`

## Next Restart

- primary: `round7`
- secondary: `round10`

## Queue

| Priority | Candidate | Bucket | Action |
| ---: | --- | --- | --- |
| `1` | `round7` | `default_restart` | `run_first` |
| `2` | `round10` | `validated_alternate` | `run_second_if_needed` |
| `3` | `round5` | `hold_only` | `defer_until_restart_and_alternate_fail` |
| `4` | `door3_post5` | `retired` | `do_not_restart` |
| `4` | `post_unlock_x5` | `retired` | `do_not_restart` |

## Seed Contract

- sentinel: `{'lane': 'prospective_c', 'seed': 193, 'mode': 'track_only'}`
- ranking_support: `{'lane': 'prospective_f', 'seed': 233, 'required_min_success': 1.0}`
- ranking_weakness: `{'lane': 'prospective_h', 'seed': 269, 'required_strictly_above': 0.984375001}`
- guardrail: `{'lane': 'prospective_h', 'seed': 277, 'required_min_success': 1.0}`

## Promotion Rules

- rules: `{'prune_on_support_regression': True, 'prune_on_guardrail_regression': True, 'hold_on_weakness_tie': True, 'broader_dev_only_after_seed_clear': True}`

## Interpretation

- This kit is the consumable handoff artifact for the next bounded restart.
- It tells future search exactly which prior to run first, which alternate to try next, which lines to keep on hold, and which ones are retired.
- It also carries the measured seed contract and promotion rules in one place so downstream tools do not need to reconstruct them from multiple reports.
