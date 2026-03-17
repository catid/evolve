# Combined DoorKey Claim Report

- external evaluation episodes per mode: `64`

| Lane | Seed | flat_dense | recovered token_dense | KL learner-state token_dense | baseline PPO SARE | KL learner-state SARE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fresh | 23 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| fresh | 29 | 1.0 | 0.0 | 0.625 | 0.0 | 1.0 |
| fresh | 31 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| original | 7 | 1.0 | 0.703125 | 1.0 | 0.0 | 1.0 |
| original | 11 | 1.0 | 0.0 | 0.0 | 0.0 | 0.5625 |
| original | 19 | 1.0 | 1.0 | 1.0 | 0.0 | 0.578125 |

## Summary

| Variant | Mean Greedy Success | Min Greedy Success | Max Greedy Success | Complete-Seed Failures |
| --- | ---: | ---: | ---: | ---: |
| flat_dense | `1.0000` | `1.0000` | `1.0000` | `0` |
| recovered token_dense | `0.4505` | `0.0000` | `1.0000` | `3` |
| KL learner-state token_dense | `0.6042` | `0.0000` | `1.0000` | `2` |
| baseline PPO SARE | `0.0000` | `0.0000` | `0.0000` | `6` |
| KL learner-state SARE | `0.8568` | `0.5625` | `1.0000` | `0` |

## Interpretation

- On the original matched lane, KL learner-state SARE mean greedy success is `0.7135` versus `0.6667` for KL learner-state token_dense.
- On the fresh matched lane, KL learner-state SARE mean greedy success is `1.0000` versus `0.5417` for KL learner-state token_dense.
- The combined six-seed DoorKey picture strengthens the routed edge: KL learner-state SARE stays ahead of the matched teacher-guided token_dense control without any complete-seed greedy failures.
