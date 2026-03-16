# Policy Extraction Report

- episodes per mode: `32`
- run count: `4`

## Greedy vs Best Sampled

| Run | Greedy Success | Best Sampled Success | Best Sampled Mode | Greedy Max Prob | Greedy Margin | Best Sampled Greedy-Match |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| minigrid_doorkey_flat_dense_ent1e3 | 1.0000 | 1.0000 | sampled_t1.0 | 0.9932 | 6.6372 | 0.9953 |
| minigrid_doorkey_sare_ent1e3 | 0.0000 | 1.0000 | sampled_t1.0 | 0.4132 | 0.4033 | 0.5076 |
| minigrid_doorkey_token_dense | 0.0000 | 0.1250 | sampled_t0.5 | 0.3153 | 0.3297 | 0.5692 |
| minigrid_doorkey_token_dense_ent1e3 | 0.7500 | 1.0000 | sampled_t1.0 | 0.7347 | 1.7735 | 0.8199 |

## Mode Table

| Run | Mode | Eval Return | Eval Success | Eval Entropy | Eval Max Prob | Eval Margin | Eval Greedy Match | Train Return | Train Success | Train Entropy | Train Max Prob | Throughput |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| minigrid_doorkey_flat_dense_ent1e3 | greedy | 0.9613 | 1.0000 | 0.0396 | 0.9932 | 6.6372 | 1.0000 | 0.9602 | 1.0000 | 0.0446 | 0.9917 | 9075.1123 |
| minigrid_doorkey_flat_dense_ent1e3 | sampled_t0.5 | 0.9613 | 1.0000 | 0.0017 | 0.9997 | 6.6372 | 1.0000 | 0.9602 | 1.0000 | 0.0446 | 0.9917 | 9075.1123 |
| minigrid_doorkey_flat_dense_ent1e3 | sampled_t0.7 | 0.9613 | 1.0000 | 0.0073 | 0.9987 | 6.6372 | 1.0000 | 0.9602 | 1.0000 | 0.0446 | 0.9917 | 9075.1123 |
| minigrid_doorkey_flat_dense_ent1e3 | sampled_t1.0 | 0.9610 | 1.0000 | 0.0396 | 0.9932 | 6.6370 | 0.9953 | 0.9602 | 1.0000 | 0.0446 | 0.9917 | 9075.1123 |
| minigrid_doorkey_sare_ent1e3 | greedy | 0.0000 | 0.0000 | 1.3489 | 0.4132 | 0.4033 | 1.0000 | 0.7443 | 0.9630 | 1.1425 | 0.5273 | 5742.9775 |
| minigrid_doorkey_sare_ent1e3 | sampled_t0.5 | 0.8288 | 1.0000 | 0.8580 | 0.6558 | 0.7082 | 0.6674 | 0.7443 | 0.9630 | 1.1425 | 0.5273 | 5742.9775 |
| minigrid_doorkey_sare_ent1e3 | sampled_t0.7 | 0.8216 | 1.0000 | 1.0417 | 0.5707 | 0.7120 | 0.5768 | 0.7443 | 0.9630 | 1.1425 | 0.5273 | 5742.9775 |
| minigrid_doorkey_sare_ent1e3 | sampled_t1.0 | 0.7697 | 1.0000 | 1.2170 | 0.4933 | 0.7095 | 0.5076 | 0.7443 | 0.9630 | 1.1425 | 0.5273 | 5742.9775 |
| minigrid_doorkey_token_dense | greedy | 0.0000 | 0.0000 | 1.7181 | 0.3153 | 0.3297 | 1.0000 | 0.3100 | 0.4444 | 1.6598 | 0.3639 | 6047.1079 |
| minigrid_doorkey_token_dense | sampled_t0.5 | 0.0820 | 0.1250 | 1.2242 | 0.5662 | 0.6035 | 0.5692 | 0.3100 | 0.4444 | 1.6598 | 0.3639 | 6047.1079 |
| minigrid_doorkey_token_dense | sampled_t0.7 | 0.0487 | 0.0938 | 1.4721 | 0.4560 | 0.5914 | 0.4566 | 0.3100 | 0.4444 | 1.6598 | 0.3639 | 6047.1079 |
| minigrid_doorkey_token_dense | sampled_t1.0 | 0.0436 | 0.0938 | 1.6611 | 0.3640 | 0.5830 | 0.3649 | 0.3100 | 0.4444 | 1.6598 | 0.3639 | 6047.1079 |
| minigrid_doorkey_token_dense_ent1e3 | greedy | 0.7219 | 0.7500 | 0.7566 | 0.7347 | 1.7735 | 1.0000 | 0.9416 | 1.0000 | 0.6950 | 0.7677 | 6022.3223 |
| minigrid_doorkey_token_dense_ent1e3 | sampled_t0.5 | 0.9540 | 1.0000 | 0.2301 | 0.9016 | 2.0174 | 0.9211 | 0.9416 | 1.0000 | 0.6950 | 0.7677 | 6022.3223 |
| minigrid_doorkey_token_dense_ent1e3 | sampled_t0.7 | 0.9515 | 1.0000 | 0.3842 | 0.8580 | 1.9797 | 0.8631 | 0.9416 | 1.0000 | 0.6950 | 0.7677 | 6022.3223 |
| minigrid_doorkey_token_dense_ent1e3 | sampled_t1.0 | 0.9460 | 1.0000 | 0.6837 | 0.7747 | 1.9865 | 0.8199 | 0.9416 | 1.0000 | 0.6950 | 0.7677 | 6022.3223 |
