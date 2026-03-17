# Updated Combined DoorKey Fairness Report

- external evaluation episodes per mode: `64`

| Lane | Seed | recovered token_dense | KL learner-state token_dense | KL learner-state single_expert | baseline PPO SARE | KL learner-state SARE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fresh | 23 | 0.0 | 0.0 | 0.40625 | 0.0 | 1.0 |
| fresh | 29 | 0.0 | 0.625 | 1.0 | 0.0 | 1.0 |
| fresh | 31 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| fresh_extra | 37 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| fresh_extra | 41 | 0.0 | 0.0 | 0.4375 | 0.0 | 1.0 |
| fresh_extra | 43 | 0.0 | 0.0 | 1.0 | 0.0 | 0.46875 |
| fresh_final | 47 | 1.0 | 1.0 | 0.453125 | 0.0 | 0.0 |
| fresh_final | 53 | 1.0 | 1.0 | 0.515625 | 0.0 | 0.515625 |
| fresh_final | 59 | 1.0 | 1.0 | 0.421875 | 0.0 | 0.421875 |
| original | 7 | 0.703125 | 1.0 | 1.0 | 0.0 | 1.0 |
| original | 11 | 0.0 | 0.0 | 1.0 | 0.0 | 0.5625 |
| original | 19 | 1.0 | 1.0 | 0.0 | 0.0 | 0.578125 |

## Summary

| Variant | Seeds Covered | Mean Greedy Success | Min Greedy Success | Max Greedy Success | Complete-Seed Failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| recovered token_dense | `12` | `0.5586` | `0.0000` | `1.0000` | `5` |
| KL learner-state token_dense | `12` | `0.6354` | `0.0000` | `1.0000` | `4` |
| KL learner-state single_expert | `12` | `0.6862` | `0.0000` | `1.0000` | `1` |
| baseline PPO SARE | `12` | `0.0000` | `0.0000` | `0.0000` | `12` |
| KL learner-state SARE | `12` | `0.7122` | `0.0000` | `1.0000` | `1` |

## Interpretation

- After adding the final-block single_expert control, KL learner-state SARE still leads slightly overall on the combined DoorKey picture, but the final block is weaker and the claim remains frozen.
