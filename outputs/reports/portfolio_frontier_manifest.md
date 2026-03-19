# Portfolio Frontier Manifest

- source campaign: `lss_portfolio_campaign`
- git commit: `fdef6f8234fbf02aa6bd15c6ed1e06a2e133483b`
- git dirty: `True`
- active benchmark pack: `outputs/reports/portfolio_candidate_pack.json`

## Frontier Roles

| Candidate | Seed-Pack Scorer | Frontier Replay | Manifest Bucket |
| --- | --- | --- | --- |
| `round6` | `n/a` | `n/a` | `active_benchmark` |
| `round7` | `advance_for_broader_dev` | `n/a` | `default_restart_prior` |
| `round10` | `advance_for_broader_dev` | `advance_for_broader_dev` | `replay_validated_alternate` |
| `round5` | `hold_seed_clean_but_below_incumbent` | `n/a` | `hold_only_prior` |
| `door3_post5` | `prune_support_regression` | `n/a` | `retired_prior` |
| `post_unlock_x5` | `prune_guardrail_regression` | `n/a` | `retired_prior` |

## Manifest Summary

- active benchmark: `['round6']`
- default restart prior: `['round7']`
- replay-validated alternates: `['round10']`
- hold-only priors: `['round5']`
- retired priors: `['door3_post5', 'post_unlock_x5']`
- seed-clean but replay-unconfirmed alternates: `[]`

## Interpretation

- `round6` remains the active benchmark.
- `round7` stays the default restart prior because it is the clean lowest-friction restart choice on the measured frontier.
- `round10` is the only replay-validated alternate that survives the historically stitched support-plus-weakness surface, so it is the measured escalation target if the project wants a fuller alternate check.
- `round5` stays hold-only: it is seed-clean on the packed frontier but still below the incumbent on broader dev mean.
- `door3_post5` and `post_unlock_x5` remain retired priors.
