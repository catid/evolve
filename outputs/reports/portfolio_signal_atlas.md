# Portfolio Signal Atlas

- source campaign: `lss_portfolio_campaign`
- git commit: `be92f7fed7ea227f3ab4397061d7ffcd0a8ccf33`
- git dirty: `True`
- dev families: `['prospective_c', 'prospective_d', 'prospective_f']`
- holdout families: `['prospective_g', 'prospective_h', 'prospective_i', 'prospective_j']`
- healthy families: `['original', 'fresh', 'fresh_extra']`

## Classification Counts

- overall counts: `{'shared_control_parity': 22, 'global_hard_failure': 3, 'benchmark_support': 4, 'control_advantaged': 1}`
- counts by group: `{'dev': {'shared_control_parity': 7, 'global_hard_failure': 1, 'benchmark_support': 1}, 'holdout': {'global_hard_failure': 2, 'shared_control_parity': 9, 'control_advantaged': 1}, 'healthy': {'shared_control_parity': 6, 'benchmark_support': 3}}`

## Seed Atlas

| Group | Lane | Seed | SARE | token_dense | single_expert | SARE-token | SARE-single | Classification |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `dev` | `prospective_c` | 181 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `dev` | `prospective_c` | 191 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `dev` | `prospective_c` | 193 | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `global_hard_failure` |
| `dev` | `prospective_d` | 197 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `dev` | `prospective_d` | 199 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `dev` | `prospective_d` | 211 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `dev` | `prospective_f` | 233 | `1.0000` | `0.6250` | `1.0000` | `0.3750` | `0.0000` | `benchmark_support` |
| `dev` | `prospective_f` | 239 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `dev` | `prospective_f` | 241 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `holdout` | `prospective_g` | 251 | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `global_hard_failure` |
| `holdout` | `prospective_g` | 257 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `holdout` | `prospective_g` | 263 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `holdout` | `prospective_h` | 269 | `0.9844` | `1.0000` | `1.0000` | `-0.0156` | `-0.0156` | `control_advantaged` |
| `holdout` | `prospective_h` | 271 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `holdout` | `prospective_h` | 277 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `holdout` | `prospective_i` | 281 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `holdout` | `prospective_i` | 283 | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `global_hard_failure` |
| `holdout` | `prospective_i` | 293 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `holdout` | `prospective_j` | 307 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `holdout` | `prospective_j` | 311 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `holdout` | `prospective_j` | 313 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `healthy` | `original` | 7 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `healthy` | `original` | 11 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `healthy` | `original` | 19 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `healthy` | `fresh` | 23 | `1.0000` | `0.5000` | `1.0000` | `0.5000` | `0.0000` | `benchmark_support` |
| `healthy` | `fresh` | 29 | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `1.0000` | `benchmark_support` |
| `healthy` | `fresh` | 31 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `healthy` | `fresh_extra` | 37 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `healthy` | `fresh_extra` | 41 | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `shared_control_parity` |
| `healthy` | `fresh_extra` | 43 | `1.0000` | `0.4688` | `1.0000` | `0.5312` | `0.0000` | `benchmark_support` |

## Informative Seeds Only

| Group | Lane | Seed | Classification | SARE | token_dense | single_expert |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `dev` | `prospective_f` | 233 | `benchmark_support` | `1.0000` | `0.6250` | `1.0000` |
| `holdout` | `prospective_h` | 269 | `control_advantaged` | `0.9844` | `1.0000` | `1.0000` |
| `healthy` | `fresh` | 23 | `benchmark_support` | `1.0000` | `0.5000` | `1.0000` |
| `healthy` | `fresh` | 29 | `benchmark_support` | `1.0000` | `1.0000` | `0.0000` |
| `healthy` | `fresh_extra` | 43 | `benchmark_support` | `1.0000` | `0.4688` | `1.0000` |

## Interpretation

- `global_hard_failure` seeds are poor challenger discriminators because every compared line is failing together.
- `shared_control_parity` seeds tell you the family is competent there, but they do not separate tied challengers from the incumbent.
- `benchmark_support` seeds are the useful support set for the active benchmark: they show where `round6` is at least as good as the controls and materially above at least one of them.
- `control_advantaged` seeds are the bounded weakness set for future work: they show where matched controls still retain an advantage and therefore deserve targeted mechanism work if the project reopens challenger search.
