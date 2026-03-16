# Learner-State Supervision Report

## Summary

| Run | Teacher | Student | Target | Before Greedy | Best Round Greedy | Final Greedy | Best Sampled After | Rounds |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `flat_dense_to_sare_lss` | flat_dense | sare | policy_head_plus_last_shared | 0.000 | 0.500 | 0.500 | 0.531 | 4 |
| `flat_dense_to_token_dense_lss` | flat_dense | token_dense | policy_head_plus_last_shared | 0.000 | 0.000 | 0.000 | 0.000 | 4 |

## Interpretation

- `flat_dense_to_sare_lss` uses learner-state supervision from `flat_dense` into `sare`, moving greedy success from `0.000` to best-round `0.500` and final `0.500`.
- `flat_dense_to_token_dense_lss` uses learner-state supervision from `flat_dense` into `token_dense`, moving greedy success from `0.000` to best-round `0.000` and final `0.000`.
