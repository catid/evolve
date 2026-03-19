# Portfolio Frontier Schedule

- source campaign: `lss_portfolio_campaign`
- git commit: `dc2fd9d4e46d7865b2927425be47ee296d1b46f4`
- git dirty: `True`
- active benchmark: `round6`

## Ordered Queue

| Priority | Candidate | Bucket | Action | Reason |
| ---: | --- | --- | --- | --- |
| `1` | `round7` | `default_restart` | `run_first` | lowest-friction measured restart prior |
| `2` | `round10` | `validated_alternate` | `run_second_if_needed` | only replay-validated alternate on the historically stitched surface |
| `3` | `round5` | `hold_only` | `defer_until_restart_and_alternate_fail` | seed-clean but below incumbent on broader dev mean |
| `4` | `door3_post5` | `retired` | `do_not_restart` | measured support or guardrail regression |
| `4` | `post_unlock_x5` | `retired` | `do_not_restart` | measured support or guardrail regression |

## Measured Seed Contract

- sentinel: `{'lane': 'prospective_c', 'seed': 193, 'use': 'track_only'}`
- ranking_support: `{'lane': 'prospective_f', 'seed': 233, 'required_min_success': 1.0}`
- ranking_weakness: `{'lane': 'prospective_h', 'seed': 269, 'required_strictly_above': 0.984375001}`
- guardrail: `{'lane': 'prospective_h', 'seed': 277, 'required_min_success': 1.0}`

## Stop Rules

- prune immediately on support regression below `1.0000` at `prospective_f/233`
- prune immediately on guardrail regression below `1.0000` at `prospective_h/277`
- do not advance lines that only tie the incumbent on `prospective_h/269`
- only spend broader dev/fairness budget after clearing the measured seed contract
