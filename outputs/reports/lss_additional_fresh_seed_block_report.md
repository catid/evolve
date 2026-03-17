# Additional Fresh DoorKey Seed Block Report

- lane: `fresh_extra`
- external evaluation episodes per mode: `64`

| Seed | flat_dense | recovered token_dense | KL learner-state token_dense | baseline PPO SARE | KL learner-state SARE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 37 | 1.0 | 1.0 | 1.0000 | 0.0 | 1.0 |
| 41 | 1.0 | 0.0 | 0.0000 | 0.0 | 1.0 |
| 43 | 1.0 | 0.0 | 0.0000 | 0.0 | 0.46875 |

## Mean Greedy Success

| Variant | Mean Greedy Success |
| --- | ---: |
| recovered token_dense | `0.3333` |
| KL learner-state token_dense | `0.3333` |
| baseline PPO SARE | `0.0000` |
| KL learner-state SARE | `0.8229` |

## Interpretation

- On the additional fresh matched DoorKey block, KL learner-state SARE still stays ahead of the matched teacher-guided token_dense control without a complete-seed routed failure.
