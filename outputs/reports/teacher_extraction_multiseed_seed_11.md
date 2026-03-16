# Policy Extraction Report

- episodes per mode: `64`
- run count: `2`

## Greedy vs Best Sampled

| Run | Greedy Success | Best Sampled Success | Best Sampled Mode | Greedy Max Prob | Greedy Margin | Best Sampled Greedy-Match |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| flat_dense_to_sare_lss | 1.0000 | 1.0000 | sampled_t1.0 | 0.9861 | 7.0549 | 0.9900 |
| token_dense_ent1e3 | 0.0000 | 0.0000 | sampled_t1.0 | 0.5371 | 1.1742 | 0.5413 |

## Mode Table

| Run | Mode | Eval Return | Eval Success | Eval Entropy | Eval Max Prob | Eval Margin | Eval Greedy Match | Train Return | Train Success | Train Entropy | Train Max Prob | Throughput |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat_dense_to_sare_lss | greedy | 0.9650 | 1.0000 | 0.0601 | 0.9861 | 7.0549 | 1.0000 | - | - | - | - | - |
| flat_dense_to_sare_lss | sampled_t0.5 | 0.9650 | 1.0000 | 0.0063 | 0.9984 | 7.0549 | 1.0000 | - | - | - | - | - |
| flat_dense_to_sare_lss | sampled_t0.7 | 0.9650 | 1.0000 | 0.0198 | 0.9956 | 7.0549 | 1.0000 | - | - | - | - | - |
| flat_dense_to_sare_lss | sampled_t1.0 | 0.9642 | 1.0000 | 0.0628 | 0.9851 | 7.0052 | 0.9900 | - | - | - | - | - |
| token_dense_ent1e3 | greedy | 0.0000 | 0.0000 | 1.3806 | 0.5371 | 1.1742 | 1.0000 | 0.0000 | 0.0000 | 1.3810 | 0.5366 | 6038.6055 |
| token_dense_ent1e3 | sampled_t0.5 | 0.0000 | 0.0000 | 0.6219 | 0.8381 | 1.1889 | 0.8402 | 0.0000 | 0.0000 | 1.3810 | 0.5366 | 6038.6055 |
| token_dense_ent1e3 | sampled_t0.7 | 0.0000 | 0.0000 | 1.0148 | 0.6946 | 1.1815 | 0.6923 | 0.0000 | 0.0000 | 1.3810 | 0.5366 | 6038.6055 |
| token_dense_ent1e3 | sampled_t1.0 | 0.0000 | 0.0000 | 1.3774 | 0.5388 | 1.1839 | 0.5413 | 0.0000 | 0.0000 | 1.3810 | 0.5366 | 6038.6055 |
