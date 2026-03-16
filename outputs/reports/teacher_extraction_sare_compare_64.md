# Policy Extraction Report

- episodes per mode: `64`
- run count: `2`

## Greedy vs Best Sampled

| Run | Greedy Success | Best Sampled Success | Best Sampled Mode | Greedy Max Prob | Greedy Margin | Best Sampled Greedy-Match |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| flat_dense_to_sare_lss | 0.5000 | 0.5625 | sampled_t1.0 | 0.9920 | 9.2367 | 0.9869 |
| minigrid_doorkey_sare_ent1e3 | 0.0000 | 1.0000 | sampled_t1.0 | 0.4177 | 0.4066 | 0.5074 |

## Mode Table

| Run | Mode | Eval Return | Eval Success | Eval Entropy | Eval Max Prob | Eval Margin | Eval Greedy Match | Train Return | Train Success | Train Entropy | Train Max Prob | Throughput |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat_dense_to_sare_lss | greedy | 0.4824 | 0.5000 | 0.0251 | 0.9920 | 9.2367 | 1.0000 | - | - | - | - | - |
| flat_dense_to_sare_lss | sampled_t0.5 | 0.4975 | 0.5156 | 0.0067 | 0.9986 | 9.2418 | 0.9984 | - | - | - | - | - |
| flat_dense_to_sare_lss | sampled_t0.7 | 0.4972 | 0.5156 | 0.0148 | 0.9960 | 9.2260 | 0.9959 | - | - | - | - | - |
| flat_dense_to_sare_lss | sampled_t1.0 | 0.5417 | 0.5625 | 0.0310 | 0.9901 | 9.1988 | 0.9869 | - | - | - | - | - |
| minigrid_doorkey_sare_ent1e3 | greedy | 0.0000 | 0.0000 | 1.3378 | 0.4177 | 0.4066 | 1.0000 | 0.7443 | 0.9630 | 1.1425 | 0.5273 | 5742.9775 |
| minigrid_doorkey_sare_ent1e3 | sampled_t0.5 | 0.8082 | 1.0000 | 0.8643 | 0.6519 | 0.6937 | 0.6534 | 0.7443 | 0.9630 | 1.1425 | 0.5273 | 5742.9775 |
| minigrid_doorkey_sare_ent1e3 | sampled_t0.7 | 0.8072 | 0.9844 | 1.0260 | 0.5749 | 0.7215 | 0.5718 | 0.7443 | 0.9630 | 1.1425 | 0.5273 | 5742.9775 |
| minigrid_doorkey_sare_ent1e3 | sampled_t1.0 | 0.7849 | 1.0000 | 1.2060 | 0.4995 | 0.7309 | 0.5074 | 0.7443 | 0.9630 | 1.1425 | 0.5273 | 5742.9775 |
