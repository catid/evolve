# Matched Teacher-Guided Control Report

- external evaluation episodes per mode: `64`

| Seed | flat_dense | recovered token_dense | KL learner-state token_dense | baseline PPO SARE | KL learner-state SARE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 | 1.0000 | 0.7031 | 1.0000 | 0.0000 | 1.0000 |
| 11 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5625 |
| 19 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5781 |

## Mean Greedy Success

| Variant | Mean Greedy Success |
| --- | ---: |
| recovered token_dense | `0.5677` |
| KL learner-state token_dense | `0.6667` |
| baseline PPO SARE | `0.0000` |
| KL learner-state SARE | `0.7135` |

## Interpretation

- Teacher-guided KL learner-state supervision helps both tokenized and routed students under matched settings.
- KL learner-state SARE still outperforms the matched teacher-guided token_dense control on mean greedy success.
