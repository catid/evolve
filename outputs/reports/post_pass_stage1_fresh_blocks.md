# Post-PASS Stage 1 Fresh Blocks

| Block | Seed | Variant | Greedy Success | Sampled t=1.0 Success | Route Entropy | Path Entropy | Active Compute |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| post_pass_a | 61 | baseline PPO SARE | 0.0000 | 0.9531 | 1.3808 | 0.9880 | 0.5000 |
| post_pass_a | 61 | KL learner-state SARE | 1.0000 | 1.0000 | 1.3702 | 0.7487 | 0.5000 |
| post_pass_a | 61 | KL learner-state single_expert | 0.0000 | 0.9688 | 0.0000 | 0.0000 | 1.0000 |
| post_pass_a | 61 | KL learner-state token_dense | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| post_pass_a | 61 | recovered token_dense | 0.0000 | 0.5625 | 0.0000 | 0.0000 | 1.0000 |
| post_pass_a | 67 | baseline PPO SARE | 0.0000 | 0.0000 | 1.3854 | 0.6847 | 0.5000 |
| post_pass_a | 67 | KL learner-state SARE | 1.0000 | 1.0000 | 1.3772 | 0.5965 | 0.5000 |
| post_pass_a | 67 | KL learner-state single_expert | 0.5781 | 0.9844 | 0.0000 | 0.0000 | 1.0000 |
| post_pass_a | 67 | KL learner-state token_dense | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| post_pass_a | 67 | recovered token_dense | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| post_pass_a | 71 | baseline PPO SARE | 0.0000 | 0.9688 | 1.3809 | 0.4134 | 0.5000 |
| post_pass_a | 71 | KL learner-state SARE | 1.0000 | 1.0000 | 1.3783 | 0.6226 | 0.5000 |
| post_pass_a | 71 | KL learner-state single_expert | 0.6250 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| post_pass_a | 71 | KL learner-state token_dense | 0.9531 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| post_pass_a | 71 | recovered token_dense | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| post_pass_b | 73 | baseline PPO SARE | 0.0000 | 1.0000 | 1.3837 | 0.6320 | 0.5000 |
| post_pass_b | 73 | KL learner-state SARE | 1.0000 | 1.0000 | 1.3778 | 0.7567 | 0.5000 |
| post_pass_b | 73 | KL learner-state single_expert | 0.0000 | 0.6094 | 0.0000 | 0.0000 | 1.0000 |
| post_pass_b | 73 | KL learner-state token_dense | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| post_pass_b | 73 | recovered token_dense | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| post_pass_b | 79 | baseline PPO SARE | 0.0000 | 0.9688 | 1.3857 | 1.2085 | 0.5000 |
| post_pass_b | 79 | KL learner-state SARE | 1.0000 | 1.0000 | 1.3585 | 1.2437 | 0.5000 |
| post_pass_b | 79 | KL learner-state single_expert | 0.0000 | 0.6875 | 0.0000 | 0.0000 | 1.0000 |
| post_pass_b | 79 | KL learner-state token_dense | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| post_pass_b | 79 | recovered token_dense | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| post_pass_b | 83 | baseline PPO SARE | 0.0000 | 0.0000 | 1.3817 | 0.1575 | 0.5000 |
| post_pass_b | 83 | KL learner-state SARE | 0.5625 | 0.6250 | 1.3608 | 0.7533 | 0.5000 |
| post_pass_b | 83 | KL learner-state single_expert | 0.5625 | 0.9844 | 0.0000 | 0.0000 | 1.0000 |
| post_pass_b | 83 | KL learner-state token_dense | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| post_pass_b | 83 | recovered token_dense | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |

## Block Summary

| Block | candidate SARE Mean | matched token_dense Mean | matched single_expert Mean | candidate - single_expert | candidate - token_dense | Complete-Seed Failures | Stage 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| post_pass_a | 1.0000 | 0.9844 | 0.4010 | 0.5990 | 0.0156 | 0 | `pass` |
| post_pass_b | 0.8542 | 1.0000 | 0.1875 | 0.6667 | -0.1458 | 0 | `pass` |

## Interpretation

- selected strong fresh-block cases for later route validation: `[('post_pass_a', 61), ('post_pass_b', 73)]`
- stage-1 status: `pass`
- The new fresh blocks use matched post_unlock_weighted learner-state runs for `token_dense`, `single_expert`, and `SARE`, plus recovered token_dense and baseline PPO SARE for context.
