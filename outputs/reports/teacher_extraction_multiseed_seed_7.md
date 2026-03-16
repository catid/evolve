# Policy Extraction Report

- episodes per mode: `64`
- run count: `2`

## Greedy vs Best Sampled

| Run | Greedy Success | Best Sampled Success | Best Sampled Mode | Greedy Max Prob | Greedy Margin | Best Sampled Greedy-Match |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| flat_dense_to_sare_lss | 0.5000 | 0.5625 | sampled_t1.0 | 0.9920 | 9.2367 | 0.9869 |
| token_dense_ent1e3 | 0.7031 | 1.0000 | sampled_t1.0 | 0.7291 | 1.7313 | 0.7937 |

## Mode Table

| Run | Mode | Eval Return | Eval Success | Eval Entropy | Eval Max Prob | Eval Margin | Eval Greedy Match | Train Return | Train Success | Train Entropy | Train Max Prob | Throughput |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat_dense_to_sare_lss | greedy | 0.4824 | 0.5000 | 0.0251 | 0.9920 | 9.2367 | 1.0000 | - | - | - | - | - |
| flat_dense_to_sare_lss | sampled_t0.5 | 0.4975 | 0.5156 | 0.0067 | 0.9986 | 9.2418 | 0.9984 | - | - | - | - | - |
| flat_dense_to_sare_lss | sampled_t0.7 | 0.4972 | 0.5156 | 0.0148 | 0.9960 | 9.2260 | 0.9959 | - | - | - | - | - |
| flat_dense_to_sare_lss | sampled_t1.0 | 0.5417 | 0.5625 | 0.0310 | 0.9901 | 9.1988 | 0.9869 | - | - | - | - | - |
| token_dense_ent1e3 | greedy | 0.6764 | 0.7031 | 0.7673 | 0.7291 | 1.7313 | 1.0000 | 0.9416 | 1.0000 | 0.6950 | 0.7677 | 6013.4028 |
| token_dense_ent1e3 | sampled_t0.5 | 0.9506 | 1.0000 | 0.2317 | 0.9047 | 2.0112 | 0.9215 | 0.9416 | 1.0000 | 0.6950 | 0.7677 | 6013.4028 |
| token_dense_ent1e3 | sampled_t0.7 | 0.9484 | 1.0000 | 0.3909 | 0.8545 | 1.9602 | 0.8606 | 0.9416 | 1.0000 | 0.6950 | 0.7677 | 6013.4028 |
| token_dense_ent1e3 | sampled_t1.0 | 0.9436 | 1.0000 | 0.6841 | 0.7741 | 1.9820 | 0.7937 | 0.9416 | 1.0000 | 0.6950 | 0.7677 | 6013.4028 |
