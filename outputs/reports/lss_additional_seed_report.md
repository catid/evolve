# Additional-Seed DoorKey Report

- external evaluation episodes per mode: `64`

| Seed | flat_dense | recovered token_dense | baseline PPO SARE | KL learner-state SARE |
| --- | ---: | ---: | ---: | ---: |
| 23 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| 29 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| 31 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |

## Mean Greedy Success

| Variant | Mean Greedy Success |
| --- | ---: |
| flat_dense | `1.0000` |
| recovered token_dense | `0.3333` |
| baseline PPO SARE | `0.0000` |
| KL learner-state SARE | `1.0000` |

## Verdict

- On the additional seed set, KL learner-state SARE stays competitive with recovered token_dense and avoids complete-seed failure.
