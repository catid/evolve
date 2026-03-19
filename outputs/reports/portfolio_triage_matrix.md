# Portfolio Triage Matrix

- source campaign: `lss_portfolio_campaign`
- git commit: `4cb03c547736ac56436f6a960b301c71dc84cdfc`
- git dirty: `True`
- target seeds: `['prospective_c/193:global_hard_sentinel', 'prospective_f/233:benchmark_support', 'prospective_h/269:bounded_weakness', 'prospective_h/277:same_lane_guardrail']`

## Active Controls On The Target Seeds

| Variant | prospective_c/193 | prospective_f/233 | prospective_h/269 | prospective_h/277 |
| --- | ---: | ---: | ---: | ---: |
| `round6` | `0.0000` | `1.0000` | `0.9844` | `1.0000` |
| `token_dense` | `0.0000` | `0.6250` | `1.0000` | `1.0000` |
| `single_expert` | `0.0000` | `1.0000` | `1.0000` | `1.0000` |

## Prior Matrix

| Candidate | Source | c/193 | f/233 | h/269 | h/277 | delta vs round6 on 269 | Bucket |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `round6` | `active` | `0.0000` | `1.0000` | `0.9844` | `1.0000` | `0.0000` | `not_better_than_incumbent` |
| `round5` | `mixed_round_prior` | `0.0000` | `1.0000` | `1.0000` | `1.0000` | `0.0156` | `conservative_clean_prior` |
| `round7` | `mixed_round_prior` | `0.0000` | `1.0000` | `1.0000` | `1.0000` | `0.0156` | `conservative_clean_prior` |
| `round10` | `mixed_round_prior` | `0.0000` | `1.0000` | `1.0000` | `1.0000` | `0.0156` | `conservative_clean_prior` |
| `door3_post5` | `migration_only` | `n/a` | `n/a` | `1.0000` | `1.0000` | `0.0156` | `support_unmeasured_structural` |
| `post_unlock_x5` | `migration_only` | `n/a` | `n/a` | `1.0000` | `0.8906` | `0.0156` | `local_only_fix` |

## Search Use

- conservative clean priors: `['round5', 'round7', 'round10']`
- structural clean priors: `[]`
- partial guardrail-loss priors: `[]`
- support-unmeasured but same-lane-clean priors: `['door3_post5']`
- local-only fixes: `['post_unlock_x5']`
- not better than incumbent on the weakness seed: `['round6']`

## Interpretation

- Every shortlisted prior still fails the global-hard sentinel `prospective_c/193`, so that seed remains a hardness sentinel rather than a near-term optimization target.
- The useful separator is the pair `prospective_f/233` plus `prospective_h/269/277`: good priors preserve the benchmark-support seed when it is measured, improve the bounded weakness seed above `round6`, and avoid degrading the same-lane guardrail.
- On that measured triage surface, `round5`, `round7`, and `round10` are the clean conservative starting points; `door3_post5` remains a same-lane-clean structural fallback but still needs direct `233` measurement before being treated as equally safe; and `post_unlock_x5` is only a partial revisit because it gives back some of the `277` guardrail.
