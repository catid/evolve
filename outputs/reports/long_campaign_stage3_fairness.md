# Long Campaign Stage 3 Fairness

| Candidate | Variant | Seed | Final Greedy | Best Round | Best Round Greedy |
| --- | --- | --- | ---: | ---: | ---: |
| `post_unlock_weighted` | KL learner-state SARE | 47 | 0.4531 | 4 | 0.4531 |
| `post_unlock_weighted` | KL learner-state SARE | 53 | 0.5156 | 3 | 0.5156 |
| `post_unlock_weighted` | KL learner-state SARE | 59 | 0.4219 | 4 | 0.4219 |
| `post_unlock_weighted` | KL learner-state single_expert | 47 | 0.4531 | 4 | 0.4531 |
| `post_unlock_weighted` | KL learner-state single_expert | 53 | 0.5156 | 3 | 0.5156 |
| `post_unlock_weighted` | KL learner-state single_expert | 59 | 0.4219 | 4 | 0.4219 |
| `post_unlock_weighted` | KL learner-state token_dense | 47 | 1.0000 | 1 | 1.0000 |
| `post_unlock_weighted` | KL learner-state token_dense | 53 | 1.0000 | 1 | 1.0000 |
| `post_unlock_weighted` | KL learner-state token_dense | 59 | 1.0000 | 1 | 1.0000 |

## Candidate Summary

| Candidate | SARE Mean | token_dense Mean | single_expert Mean | SARE Delta | token Delta | single Delta | Stage 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `post_unlock_weighted` | 0.4635 | 1.0000 | 0.4635 | 0.1510 | 0.0000 | 0.0000 | `pass` |

## Interpretation

- stage-3 survivors: `['post_unlock_weighted']`
- best surviving candidate: `post_unlock_weighted`
