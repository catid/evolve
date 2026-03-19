# Portfolio Restart Policy

- source campaign: `lss_portfolio_campaign`
- git commit: `78566fd86c436424f09dfef71cfa526b3bfe6ea1`
- git dirty: `True`
- stage3 surviving challengers: `[]`
- structural support status: `measured_support_regression`

## Restart Frontier

| Candidate | Support 233 | Weakness 269 | Guardrail 277 | Triage Bucket | Dossier Bucket | Policy Bucket |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `round6` | `1.0000` | `0.9844` | `1.0000` | `not_better_than_incumbent` | `n/a` | `active_incumbent` |
| `round7` | `1.0000` | `1.0000` | `1.0000` | `conservative_clean_prior` | `recommended_conservative_default` | `restart_default` |
| `round10` | `1.0000` | `1.0000` | `1.0000` | `conservative_clean_prior` | `higher_cost_same_signal` | `reserve_same_signal_higher_cost` |
| `round5` | `1.0000` | `1.0000` | `1.0000` | `conservative_clean_prior` | `cheapest_but_below_incumbent` | `reserve_below_incumbent` |
| `door3_post5` | `0.4531` | `1.0000` | `1.0000` | `support_unmeasured_structural` | `n/a` | `retire_structural_regression` |
| `post_unlock_x5` | `n/a` | `1.0000` | `0.8906` | `local_only_fix` | `n/a` | `retire_local_only_fix` |

## Search Policy

- active incumbent: `['round6']`
- bounded restart default: `['round7']`
- reserve priors: `['round10', 'round5']`
- retired priors: `['door3_post5', 'post_unlock_x5']`

## Interpretation

- The post-control field is still empty, so future bounded search should restart from measured priors rather than pretend a live challenger already exists.
- `round7` is the only clean restart default: it preserves the measured support seed, fixes the bounded weakness seed, preserves the guardrail seed, and does so without the higher cost of `round10` or the below-incumbent broader dev mean of `round5`.
- `door3_post5` should be retired as a fallback prior, not kept in reserve. The direct probe measured a support regression on `prospective_f/233`, so it is no longer merely unmeasured.
- `post_unlock_x5` should also stay retired as a local-only fix because it gives back guardrail performance on `prospective_h/277` even though it can solve the bounded weakness seed.
