# Learner-State Robustness Multi-Seed Report

- external evaluation episodes per mode: `64`

| Seed | flat_dense | recovered token_dense | baseline SARE | improved learner-state SARE |
| --- | ---: | ---: | ---: | ---: |
| 7 | 1.0000 | 0.7031 | 0.0000 | 1.0000 |
| 11 | 1.0000 | 0.0000 | 0.0000 | 0.5625 |
| 19 | 1.0000 | 1.0000 | 0.0000 | 0.5781 |

## Mean Greedy Success

| Variant | Mean Greedy Success |
| --- | ---: |
| flat_dense | `1.0000` |
| recovered token_dense | `0.5677` |
| baseline SARE | `0.0000` |
| improved learner-state SARE | `0.7135` |

## Verdict

- The improved learner-state method passes the repo’s reopen-routed-claim bar on the 3-seed external gate.
