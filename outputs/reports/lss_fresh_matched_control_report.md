# Fresh Matched Teacher-Guided DoorKey Control Report

- lane: `fresh`
- external evaluation episodes per mode: `64`

| Seed | flat_dense | recovered token_dense | KL learner-state token_dense | baseline PPO SARE | KL learner-state SARE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 23 | 1.0 | 0.0 | 0.0000 | 0.0 | 1.0 |
| 29 | 1.0 | 0.0 | 0.6250 | 0.0 | 1.0 |
| 31 | 1.0 | 1.0 | 1.0000 | 0.0 | 1.0 |

## Mean Greedy Success

| Variant | Mean Greedy Success |
| --- | ---: |
| recovered token_dense | `0.3333` |
| KL learner-state token_dense | `0.5417` |
| baseline PPO SARE | `0.0000` |
| KL learner-state SARE | `1.0000` |

## Interpretation

- On the fresh matched teacher-guided lane, KL learner-state SARE still holds a mean greedy-success edge over KL learner-state token_dense and avoids complete-seed failure.
