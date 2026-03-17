# Matched Teacher-Guided Single-Expert DoorKey Control Report

- external evaluation episodes per mode: `64`

| Lane | Seed | flat_dense | recovered token_dense | KL learner-state token_dense | baseline PPO SARE | KL learner-state single_expert | KL learner-state SARE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 7 | 1.0 | 0.703125 | 1.0 | 0.0 | 1.0000 | 1.0 |
| original | 11 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 0.5625 |
| original | 19 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0000 | 0.578125 |

## Summary

| Variant | Seeds Covered | Mean Greedy Success | Min Greedy Success | Max Greedy Success | Complete-Seed Failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| recovered token_dense | `3` | `0.5677` | `0.0000` | `1.0000` | `1` |
| KL learner-state token_dense | `3` | `0.6667` | `0.0000` | `1.0000` | `1` |
| baseline PPO SARE | `3` | `0.0000` | `0.0000` | `0.0000` | `3` |
| KL learner-state single_expert | `3` | `0.6667` | `0.0000` | `1.0000` | `1` |
| KL learner-state SARE | `3` | `0.7135` | `0.5625` | `1.0000` | `0` |

## Interpretation

- On the matched missing-control lane, KL learner-state SARE still stays ahead of both the matched token_dense and matched single_expert teacher-guided controls.
- This fairness fill-in currently covers the original DoorKey lane only; fresh-lane single_expert controls were not required for the bounded acceptance bar.
