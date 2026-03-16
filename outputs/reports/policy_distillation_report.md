# Policy Distillation Report

## Summary

| Run | Teacher | Student | Target | Loss | Before Greedy | After Greedy | Before Best Sampled | After Best Sampled | Harvest Episodes | Trainable Params |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `flat_dense_to_sare_full` | flat_dense | sare | full_model | ce | 0.000 | 0.000 | 1.000 | 0.938 | 64 | 346248 |
| `flat_dense_to_sare_head` | flat_dense | sare | policy_head | ce | 0.000 | 0.000 | 1.000 | 0.906 | 64 | 17671 |
| `flat_dense_to_sare_last_shared` | flat_dense | sare | policy_head_plus_last_shared | ce | 0.000 | 0.000 | 1.000 | 0.844 | 64 | 299527 |
| `token_dense_to_sare_head` | token_dense | sare | policy_head | ce | 0.000 | 0.000 | 1.000 | 0.969 | 64 | 17671 |
| `token_dense_to_sare_last_shared` | token_dense | sare | policy_head_plus_last_shared | ce | 0.000 | 0.000 | 1.000 | 1.000 | 64 | 299527 |
| `flat_dense_to_token_dense_full` | flat_dense | token_dense | full_model | ce | 0.000 | 0.000 | 0.125 | 0.969 | 64 | 461192 |
| `flat_dense_to_token_dense_head` | flat_dense | token_dense | policy_head | ce | 0.000 | 0.000 | 0.125 | 0.281 | 64 | 17671 |
| `flat_dense_to_token_dense_last_shared` | flat_dense | token_dense | policy_head_plus_last_shared | ce | 0.000 | 0.000 | 0.125 | 0.750 | 64 | 216199 |
| `token_dense_to_token_dense_head` | token_dense | token_dense | policy_head | ce | 0.000 | 0.000 | 0.125 | 0.312 | 64 | 17671 |
| `token_dense_to_token_dense_last_shared` | token_dense | token_dense | policy_head_plus_last_shared | ce | 0.000 | 0.000 | 0.125 | 0.781 | 64 | 216199 |

## Best Run By Student/Teacher Pair

| Teacher | Student | Best Target | Best Loss | Best Greedy Success | Greedy Delta | Best Sampled Success |
| --- | --- | --- | --- | ---: | ---: | ---: |
| flat_dense | sare | full_model | ce | 0.000 | 0.000 | 0.938 |
| flat_dense | token_dense | full_model | ce | 0.000 | 0.000 | 0.969 |
| token_dense | sare | policy_head_plus_last_shared | ce | 0.000 | 0.000 | 1.000 |
| token_dense | token_dense | policy_head_plus_last_shared | ce | 0.000 | 0.000 | 0.781 |

## Interpretation

- `flat_dense_to_sare_full` uses `flat_dense` as teacher and `sare` as student, moving greedy success from `0.000` to `0.000` and best sampled success from `1.000` to `0.938`.
- `flat_dense_to_sare_head` uses `flat_dense` as teacher and `sare` as student, moving greedy success from `0.000` to `0.000` and best sampled success from `1.000` to `0.906`.
- `flat_dense_to_sare_last_shared` uses `flat_dense` as teacher and `sare` as student, moving greedy success from `0.000` to `0.000` and best sampled success from `1.000` to `0.844`.
- `token_dense_to_sare_head` uses `token_dense` as teacher and `sare` as student, moving greedy success from `0.000` to `0.000` and best sampled success from `1.000` to `0.969`.
- `token_dense_to_sare_last_shared` uses `token_dense` as teacher and `sare` as student, moving greedy success from `0.000` to `0.000` and best sampled success from `1.000` to `1.000`.
- `flat_dense_to_token_dense_full` uses `flat_dense` as teacher and `token_dense` as student, moving greedy success from `0.000` to `0.000` and best sampled success from `0.125` to `0.969`.
- `flat_dense_to_token_dense_head` uses `flat_dense` as teacher and `token_dense` as student, moving greedy success from `0.000` to `0.000` and best sampled success from `0.125` to `0.281`.
- `flat_dense_to_token_dense_last_shared` uses `flat_dense` as teacher and `token_dense` as student, moving greedy success from `0.000` to `0.000` and best sampled success from `0.125` to `0.750`.
- `token_dense_to_token_dense_head` uses `token_dense` as teacher and `token_dense` as student, moving greedy success from `0.000` to `0.000` and best sampled success from `0.125` to `0.312`.
- `token_dense_to_token_dense_last_shared` uses `token_dense` as teacher and `token_dense` as student, moving greedy success from `0.000` to `0.000` and best sampled success from `0.125` to `0.781`.
