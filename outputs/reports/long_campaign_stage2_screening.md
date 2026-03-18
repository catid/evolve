# Long Campaign Stage 2 Screening

| Candidate | Seed | Final Greedy | Best Round | Best Round Greedy | Best Round Disagreement | Best Round Unique Ratio | Best Round Post-Unlock Frac | Packability |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cap_recent_4096` | 47 | 0.0000 | 1 | 0.0000 | 1.0000 | 0.0016 | 0.0000 | weak_block_fail |
| `cap_recent_4096` | 53 | 0.0000 | 1 | 0.0000 | 0.9996 | 0.0018 | 0.0000 | weak_block_fail |
| `cap_recent_4096` | 59 | 0.0000 | 1 | 0.0000 | 0.6367 | 0.0023 | 0.0000 | weak_block_fail |
| `phase_balanced_recent_4096` | 47 | 0.0000 | 1 | 0.0000 | 1.0000 | 0.0016 | 0.0000 | weak_block_fail |
| `phase_balanced_recent_4096` | 53 | 0.0000 | 1 | 0.0000 | 0.9996 | 0.0018 | 0.0000 | weak_block_fail |
| `phase_balanced_recent_4096` | 59 | 0.0000 | 1 | 0.0000 | 0.6367 | 0.0023 | 0.0000 | weak_block_fail |
| `phase_balanced_recent_4096_post_unlock_weighted` | 47 | 0.0000 | 1 | 0.0000 | 1.0000 | 0.0016 | 0.0000 | weak_block_fail |
| `phase_balanced_recent_4096_post_unlock_weighted` | 53 | 0.0000 | 1 | 0.0000 | 0.9996 | 0.0018 | 0.0000 | weak_block_fail |
| `phase_balanced_recent_4096_post_unlock_weighted` | 59 | 0.0000 | 1 | 0.0000 | 0.6367 | 0.0023 | 0.0000 | weak_block_fail |
| `post_unlock_weighted` | 47 | 0.4531 | 4 | 0.4531 | 0.9747 | 0.0033 | 0.7646 | ready |
| `post_unlock_weighted` | 53 | 0.5156 | 3 | 0.5156 | 0.3731 | 0.0029 | 0.0000 | ready |
| `post_unlock_weighted` | 59 | 0.4219 | 4 | 0.4219 | 0.9715 | 0.0032 | 0.9795 | ready |

## Candidate Summary

| Candidate | Mean Greedy Success | Complete-Seed Failures | Stage 2 |
| --- | ---: | ---: | --- |
| `post_unlock_weighted` | `0.4635` | `0` | `pass` |
| `cap_recent_4096` | `0.0000` | `3` | `stop` |
| `phase_balanced_recent_4096` | `0.0000` | `3` | `stop` |
| `phase_balanced_recent_4096_post_unlock_weighted` | `0.0000` | `3` | `stop` |

## Interpretation

- advancing candidates: `['post_unlock_weighted']`
- Stage 2 keeps only candidates that beat the frozen weak-block `SARE` mean and show a non-noisy per-seed improvement pattern on `47/53/59`.
