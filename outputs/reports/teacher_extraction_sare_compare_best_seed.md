# Policy Extraction Report

- episodes per mode: `64`
- run count: `2`

## Greedy vs Best Sampled

| Run | Greedy Success | Best Sampled Success | Best Sampled Mode | Greedy Max Prob | Greedy Margin | Best Sampled Greedy-Match |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| flat_dense_to_sare_lss | 1.0000 | 1.0000 | sampled_t1.0 | 0.9861 | 7.0549 | 0.9900 |
| sare_ent1e3 | 0.0000 | 0.7812 | sampled_t1.0 | 0.3813 | 0.3966 | 0.4359 |

## Mode Table

| Run | Mode | Eval Return | Eval Success | Eval Entropy | Eval Max Prob | Eval Margin | Eval Greedy Match | Train Return | Train Success | Train Entropy | Train Max Prob | Throughput |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat_dense_to_sare_lss | greedy | 0.9650 | 1.0000 | 0.0601 | 0.9861 | 7.0549 | 1.0000 | - | - | - | - | - |
| flat_dense_to_sare_lss | sampled_t0.5 | 0.9650 | 1.0000 | 0.0063 | 0.9984 | 7.0549 | 1.0000 | - | - | - | - | - |
| flat_dense_to_sare_lss | sampled_t0.7 | 0.9650 | 1.0000 | 0.0198 | 0.9956 | 7.0549 | 1.0000 | - | - | - | - | - |
| flat_dense_to_sare_lss | sampled_t1.0 | 0.9642 | 1.0000 | 0.0628 | 0.9851 | 7.0052 | 0.9900 | - | - | - | - | - |
| sare_ent1e3 | greedy | 0.0000 | 0.0000 | 1.4439 | 0.3813 | 0.3966 | 1.0000 | 0.4203 | 0.6316 | 1.3089 | 0.4694 | 5745.4146 |
| sare_ent1e3 | sampled_t0.5 | 0.4623 | 0.7812 | 1.0146 | 0.5905 | 0.5726 | 0.5863 | 0.4203 | 0.6316 | 1.3089 | 0.4694 | 5745.4146 |
| sare_ent1e3 | sampled_t0.7 | 0.4733 | 0.7812 | 1.1619 | 0.5205 | 0.6268 | 0.5110 | 0.4203 | 0.6316 | 1.3089 | 0.4694 | 5745.4146 |
| sare_ent1e3 | sampled_t1.0 | 0.4444 | 0.7812 | 1.3619 | 0.4384 | 0.6351 | 0.4359 | 0.4203 | 0.6316 | 1.3089 | 0.4694 | 5745.4146 |
