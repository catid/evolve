# Expanded Combined DoorKey Claim Report

- external evaluation episodes per mode: `64`

| Lane | Seed | flat_dense | recovered token_dense | KL learner-state token_dense | baseline PPO SARE | KL learner-state single_expert | KL learner-state SARE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fresh | 23 | 1.0 | 0.0 | 0.0 | 0.0 | - | 1.0 |
| fresh | 29 | 1.0 | 0.0 | 0.625 | 0.0 | - | 1.0 |
| fresh | 31 | 1.0 | 1.0 | 1.0 | 0.0 | - | 1.0 |
| fresh_extra | 37 | 1.0 | 1.0 | 1.0 | 0.0 | - | 1.0 |
| fresh_extra | 41 | 1.0 | 0.0 | 0.0 | 0.0 | - | 1.0 |
| fresh_extra | 43 | 1.0 | 0.0 | 0.0 | 0.0 | - | 0.46875 |
| original | 7 | 1.0 | 0.703125 | 1.0 | 0.0 | 1.0 | 1.0 |
| original | 11 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.5625 |
| original | 19 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.578125 |

## Summary

| Variant | Seeds Covered | Mean Greedy Success | Min Greedy Success | Max Greedy Success | Complete-Seed Failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| flat_dense | `9` | `1.0000` | `1.0000` | `1.0000` | `0` |
| recovered token_dense | `9` | `0.4115` | `0.0000` | `1.0000` | `5` |
| KL learner-state token_dense | `9` | `0.5139` | `0.0000` | `1.0000` | `4` |
| baseline PPO SARE | `9` | `0.0000` | `0.0000` | `0.0000` | `9` |
| KL learner-state single_expert | `3` | `0.6667` | `0.0000` | `1.0000` | `1` |
| KL learner-state SARE | `9` | `0.8455` | `0.4688` | `1.0000` | `0` |

## Interpretation

- Across the expanded DoorKey picture, KL learner-state SARE mean greedy success is `0.8455` versus `0.5139` for matched KL learner-state token_dense.
- On the matched missing-control slice, KL learner-state SARE mean greedy success is `0.7135` versus `0.6667` for KL learner-state single_expert.
- The expanded DoorKey picture broadens the claim within scope: KL learner-state SARE stays ahead of the matched structured controls without any complete-seed routed failure.
