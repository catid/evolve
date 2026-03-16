# Policy Extraction Report

- episodes per mode: `64`
- run count: `2`

## Greedy vs Best Sampled

| Run | Greedy Success | Best Sampled Success | Best Sampled Mode | Greedy Max Prob | Greedy Margin | Best Sampled Greedy-Match |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| flat_dense_to_sare_lss | 0.0000 | 0.0000 | sampled_t1.0 | 0.9369 | 4.8506 | 0.9974 |
| token_dense_ent1e3 | 1.0000 | 1.0000 | sampled_t1.0 | 0.9755 | 5.3730 | 0.9745 |

## Mode Table

| Run | Mode | Eval Return | Eval Success | Eval Entropy | Eval Max Prob | Eval Margin | Eval Greedy Match | Train Return | Train Success | Train Entropy | Train Max Prob | Throughput |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat_dense_to_sare_lss | greedy | 0.0000 | 0.0000 | 0.1984 | 0.9369 | 4.8506 | 1.0000 | - | - | - | - | - |
| flat_dense_to_sare_lss | sampled_t0.5 | 0.0000 | 0.0000 | 0.0135 | 0.9973 | 8.3006 | 0.9980 | - | - | - | - | - |
| flat_dense_to_sare_lss | sampled_t0.7 | 0.0000 | 0.0000 | 0.0095 | 0.9976 | 9.2761 | 0.9978 | - | - | - | - | - |
| flat_dense_to_sare_lss | sampled_t1.0 | 0.0000 | 0.0000 | 0.0120 | 0.9970 | 9.5318 | 0.9974 | - | - | - | - | - |
| token_dense_ent1e3 | greedy | 0.9639 | 1.0000 | 0.1217 | 0.9755 | 5.3730 | 1.0000 | 0.9626 | 1.0000 | 0.1441 | 0.9685 | 6045.2119 |
| token_dense_ent1e3 | sampled_t0.5 | 0.9637 | 1.0000 | 0.0102 | 0.9979 | 5.3466 | 0.9976 | 0.9626 | 1.0000 | 0.1441 | 0.9685 | 6045.2119 |
| token_dense_ent1e3 | sampled_t0.7 | 0.9630 | 1.0000 | 0.0360 | 0.9925 | 5.3158 | 0.9894 | 0.9626 | 1.0000 | 0.1441 | 0.9685 | 6045.2119 |
| token_dense_ent1e3 | sampled_t1.0 | 0.9621 | 1.0000 | 0.1298 | 0.9733 | 5.2995 | 0.9745 | 0.9626 | 1.0000 | 0.1441 | 0.9685 | 6045.2119 |
