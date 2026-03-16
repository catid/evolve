# Self-Imitation Report

## Summary

| Run | Variant | Target | Weighting | Harvest Successes | Before Greedy Success | After Greedy Success | Before Best Sampled | After Best Sampled | Trainable Params |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sare_policy_head_return` | sare | policy_head | return | 64 | 0.000 | 0.000 | 1.000 | 1.000 | 17671 |
| `sare_policy_head_uniform` | sare | policy_head | uniform | 64 | 0.000 | 0.000 | 1.000 | 1.000 | 17671 |
| `sare_policy_head_plus_last_shared_return` | sare | policy_head_plus_last_shared | return | 64 | 0.000 | 0.000 | 1.000 | 1.000 | 299527 |
| `sare_policy_head_plus_last_shared_uniform` | sare | policy_head_plus_last_shared | uniform | 64 | 0.000 | 0.000 | 1.000 | 1.000 | 299527 |
| `single_expert_policy_head_return` | single_expert | policy_head | return | 64 | 0.000 | 0.000 | 0.750 | 0.844 | 17671 |
| `single_expert_policy_head_uniform` | single_expert | policy_head | uniform | 64 | 0.000 | 0.000 | 0.750 | 0.906 | 17671 |
| `single_expert_policy_head_plus_last_shared_return` | single_expert | policy_head_plus_last_shared | return | 64 | 0.000 | 0.000 | 0.750 | 0.719 | 100615 |
| `single_expert_policy_head_plus_last_shared_uniform` | single_expert | policy_head_plus_last_shared | uniform | 64 | 0.000 | 0.000 | 0.750 | 0.750 | 100615 |

## Interpretation

- `sare_policy_head_return` moves greedy success from `0.000` to `0.000` while sampled success moves from `1.000` to `1.000`.
- `sare_policy_head_uniform` moves greedy success from `0.000` to `0.000` while sampled success moves from `1.000` to `1.000`.
- `sare_policy_head_plus_last_shared_return` moves greedy success from `0.000` to `0.000` while sampled success moves from `1.000` to `1.000`.
- `sare_policy_head_plus_last_shared_uniform` moves greedy success from `0.000` to `0.000` while sampled success moves from `1.000` to `1.000`.
- `single_expert_policy_head_return` moves greedy success from `0.000` to `0.000` while sampled success moves from `0.750` to `0.844`.
- `single_expert_policy_head_uniform` moves greedy success from `0.000` to `0.000` while sampled success moves from `0.750` to `0.906`.
- `single_expert_policy_head_plus_last_shared_return` moves greedy success from `0.000` to `0.000` while sampled success moves from `0.750` to `0.719`.
- `single_expert_policy_head_plus_last_shared_uniform` moves greedy success from `0.000` to `0.000` while sampled success moves from `0.750` to `0.750`.
