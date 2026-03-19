# Portfolio Weakness Dossier

- source campaign: `lss_portfolio_campaign`
- git commit: `6b4e078902dd4d863108a47946c76df54fd43b28`
- git dirty: `True`
- overall classification counts: `{'shared_control_parity': 22, 'global_hard_failure': 3, 'benchmark_support': 4, 'control_advantaged': 1}`
- bounded weakness seeds: `['prospective_h/269']`

## Primary Weakness

- primary weakness seed: `prospective_h/269`
- group: `holdout`
- final success: `round6=0.9844`, `token_dense=1.0000`, `single_expert=1.0000`
- final deltas: `round6-token_dense=-0.0156`, `round6-single_expert=-0.0156`
- best-round indices: `round6=4`, `token_dense=4`, `single_expert=5`

## Same-Lane Contrast

| Lane | Seed | Classification | round6 | token_dense | single_expert | round6-token | round6-single |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `prospective_h` | 269 | `control_advantaged` | `0.9844` | `1.0000` | `1.0000` | `-0.0156` | `-0.0156` |
| `prospective_h` | 271 | `shared_control_parity` | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` |
| `prospective_h` | 277 | `shared_control_parity` | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` |

## Round Trace On The Weakness Seed

| Variant | Round | Greedy | Post-Unlock Frac | Disagreement | Route Entropy | Path Entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `round6` | `1` | `0.0000` | `0.0000` | `1.0000` | `1.3845` | `0.5216` |
| `round6` | `2` | `0.0000` | `0.0000` | `0.9313` | `1.3636` | `0.7023` |
| `round6` | `3` | `0.6250` | `0.0025` | `0.6434` | `1.3677` | `0.8053` |
| `round6` | `4` | `1.0000` | `0.9572` | `0.3893` | `1.3652` | `0.8710` |
| `round6` | `5` | `1.0000` | `0.4426` | `0.0000` | `1.3653` | `0.8704` |
| `round6` | `6` | `0.9844` | `0.4418` | `0.0000` | `1.3632` | `0.8807` |
| `token_dense` | `1` | `0.8750` | `0.3773` | `0.1095` | `0.0000` | `n/a` |
| `token_dense` | `2` | `0.6406` | `0.1194` | `0.4252` | `0.0000` | `n/a` |
| `token_dense` | `3` | `0.6562` | `0.0282` | `0.4687` | `0.0000` | `n/a` |
| `token_dense` | `4` | `1.0000` | `0.0337` | `0.8288` | `0.0000` | `n/a` |
| `token_dense` | `5` | `1.0000` | `0.4426` | `0.0000` | `0.0000` | `n/a` |
| `token_dense` | `6` | `1.0000` | `0.4418` | `0.0000` | `0.0000` | `n/a` |
| `single_expert` | `1` | `0.0000` | `0.0000` | `0.9984` | `0.0000` | `0.0000` |
| `single_expert` | `2` | `0.0000` | `0.0000` | `0.9630` | `0.0000` | `0.0000` |
| `single_expert` | `3` | `0.0000` | `0.0000` | `0.4931` | `0.0000` | `0.0000` |
| `single_expert` | `4` | `0.6250` | `0.8110` | `0.6612` | `0.0000` | `0.0000` |
| `single_expert` | `5` | `1.0000` | `0.9550` | `0.4647` | `0.0000` | `0.0000` |
| `single_expert` | `6` | `1.0000` | `0.4418` | `0.0000` | `0.0000` | `0.0000` |

## Interpretation

- The active benchmark has exactly one bounded control-advantaged seed on the current DoorKey surface, so the weakness is localized rather than broad.
- On that seed, `round6` does recover to `1.0000` mid-run before finishing at `0.9844`, which makes this a late-round end-state slip rather than a full inability to solve the case.
- The neighboring holdout seeds in the same lane (`271` and `277`) still finish at full parity, which confirms the weakness is seed-local within `prospective_h` rather than lane-wide.
- Future challenger work should treat this seed as the main bounded weakness target while avoiding more spend on seeds that are already shared parity or global-hard failures.
