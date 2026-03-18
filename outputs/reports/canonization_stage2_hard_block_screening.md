# Canonization Stage 2 Hard-Block Screening

- incumbent hard-block family mean: `0.5938`
- incumbent matched hard-block token_dense mean: `0.9453`
- incumbent matched hard-block single_expert mean: `0.2057`

| Candidate | Block | Seed | Final Greedy | Δ vs Current SARE | Δ vs Current token_dense | Δ vs Current single_expert | Best Round | Best-Round Disagreement |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `post_unlock_weighted_disagreement075` | post_pass_b | 73 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 3 | 0.7274 |
| `post_unlock_weighted_disagreement075` | post_pass_b | 79 | 0.5625 | -0.4375 | -0.4375 | 0.5625 | 4 | 0.6294 |
| `post_unlock_weighted_disagreement075` | post_pass_b | 83 | 0.5625 | 0.0000 | -0.4375 | 0.0000 | 4 | 0.9796 |
| `post_unlock_weighted_disagreement075` | post_pass_c | 89 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 4 | 0.9237 |
| `post_unlock_weighted_disagreement075` | post_pass_c | 97 | 0.0000 | 0.0000 | -1.0000 | 0.0000 | 1 | 0.9954 |
| `post_unlock_weighted_disagreement075` | post_pass_c | 101 | 1.0000 | 0.0000 | 0.3281 | 0.3281 | 4 | 0.9668 |
| `post_unlock_weighted_round5` | post_pass_b | 73 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 3 | 0.7274 |
| `post_unlock_weighted_round5` | post_pass_b | 79 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 3 | 0.7038 |
| `post_unlock_weighted_round5` | post_pass_b | 83 | 0.5625 | 0.0000 | -0.4375 | 0.0000 | 4 | 0.9796 |
| `post_unlock_weighted_round5` | post_pass_c | 89 | 0.5156 | 0.5156 | -0.4844 | 0.5156 | 5 | 0.5413 |
| `post_unlock_weighted_round5` | post_pass_c | 97 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 5 | 0.9645 |
| `post_unlock_weighted_round5` | post_pass_c | 101 | 1.0000 | 0.0000 | 0.3281 | 0.3281 | 4 | 0.9668 |
| `post_unlock_weighted_x6` | post_pass_b | 73 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 3 | 0.7274 |
| `post_unlock_weighted_x6` | post_pass_b | 79 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 4 | 0.4573 |
| `post_unlock_weighted_x6` | post_pass_b | 83 | 0.0000 | -0.5625 | -1.0000 | -0.5625 | 1 | 1.0000 |
| `post_unlock_weighted_x6` | post_pass_c | 89 | 0.5156 | 0.5156 | -0.4844 | 0.5156 | 4 | 0.9709 |
| `post_unlock_weighted_x6` | post_pass_c | 97 | 0.0000 | 0.0000 | -1.0000 | 0.0000 | 1 | 0.9954 |
| `post_unlock_weighted_x6` | post_pass_c | 101 | 1.0000 | 0.0000 | 0.3281 | 0.3281 | 4 | 0.9668 |

## Candidate Summary

| Candidate | Hard-Family Mean | Δ vs Current SARE | Gap Narrowing vs token_dense | Candidate - token_dense | Candidate - single_expert | Complete-Seed Failures | Packability | Stage 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `post_unlock_weighted_round5` | 0.8464 | 0.2526 | 0.2526 | -0.0990 | 0.6406 | 0 | ready | `pass` |
| `post_unlock_weighted_disagreement075` | 0.6875 | 0.0938 | 0.0938 | -0.2578 | 0.4818 | 1 | ready | `pass` |
| `post_unlock_weighted_x6` | 0.5859 | -0.0078 | -0.0078 | -0.3594 | 0.3802 | 2 | hard_block_gap | `stop` |

## Interpretation

- advancing candidates: `['post_unlock_weighted_round5', 'post_unlock_weighted_disagreement075']`
- Stage 2 keeps only hard-block candidates that materially improve over the thaw-qualified incumbent, narrow the token_dense gap, and avoid new complete-seed failures.
