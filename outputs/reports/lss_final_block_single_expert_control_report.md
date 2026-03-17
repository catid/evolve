# Final-Block Matched Single-Expert DoorKey Control Report

- lane: `fresh_final`
- external evaluation episodes per mode: `64`

| Seed | recovered token_dense | KL learner-state token_dense | KL learner-state single_expert | baseline PPO SARE | KL learner-state SARE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 47 | 1.0 | 1.0 | 0.4531 | 0.0 | 0.0 |
| 53 | 1.0 | 1.0 | 0.5156 | 0.0 | 0.515625 |
| 59 | 1.0 | 1.0 | 0.4219 | 0.0 | 0.421875 |

## Mean Greedy Success

| Variant | Mean Greedy Success | Complete-Seed Failures |
| --- | ---: | ---: |
| recovered token_dense | `1.0000` | `0` |
| KL learner-state token_dense | `1.0000` | `0` |
| KL learner-state single_expert | `0.4635` | `0` |
| baseline PPO SARE | `0.0000` | `3` |
| KL learner-state SARE | `0.3125` | `1` |

## Interpretation

- On the final fresh block, matched structured controls catch or beat KL learner-state SARE, so the current claim should remain frozen or narrow further.
