# Run Summary

## Aggregate By Environment / Variant

### doorkey_5x5

| variant | seeds | eval_return | eval_success | best_train_return | train_eval_gap | throughput | compute_proxy | route_entropy | hop_count | rel_usage | option_duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat_dense | 1 | 0.965 | 1.000 | 0.960 | -0.005 | 9668.3 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| single_expert | 1 | 0.000 | 0.000 | 0.023 | 0.000 | 7059.0 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| token_dense | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 6587.7 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### doorkey_5x5_fullobs

| variant | seeds | eval_return | eval_success | best_train_return | train_eval_gap | throughput | compute_proxy | route_entropy | hop_count | rel_usage | option_duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| token_dense | 1 | 0.000 | 0.000 | 0.410 | 0.162 | 6456.1 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### keycorridors3r1

| variant | seeds | eval_return | eval_success | best_train_return | train_eval_gap | throughput | compute_proxy | route_entropy | hop_count | rel_usage | option_duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat_dense | 1 | 0.000 | 0.000 | 0.337 | 0.337 | 9756.0 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| single_expert | 1 | 0.000 | 0.000 | 0.687 | 0.000 | 7038.2 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| token_dense | 1 | 0.000 | 0.000 | 0.240 | 0.000 | 6533.3 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### memorys9

| variant | seeds | eval_return | eval_success | best_train_return | train_eval_gap | throughput | compute_proxy | route_entropy | hop_count | rel_usage | option_duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| token_dense | 1 | 0.000 | 0.000 | 0.561 | 0.422 | 6502.7 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| token_gru | 1 | 0.000 | 0.000 | 0.537 | 0.445 | 6260.1 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Per-Run Details

### doorkey_5x5

| variant | seed | greedy_eval | fullobs | eval_return | eval_success | final_train_return | best_train_return | train_eval_gap | explained_variance | entropy | throughput | run |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flat_dense | 7 | true | false | 0.965 | 1.000 | 0.960 | 0.960 | -0.005 | 0.440 | 0.382 | 9668.3 | `doorkey_flat_dense` |
| single_expert | 7 | true | false | 0.000 | 0.000 | 0.000 | 0.023 | 0.000 | -2.328 | 1.786 | 7059.0 | `doorkey_single_expert` |
| token_dense | 7 | true | false | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -2.966 | 1.700 | 6587.7 | `doorkey_token_dense` |

### doorkey_5x5_fullobs

| variant | seed | greedy_eval | fullobs | eval_return | eval_success | final_train_return | best_train_return | train_eval_gap | explained_variance | entropy | throughput | run |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| token_dense | 7 | true | true | 0.000 | 0.000 | 0.162 | 0.410 | 0.162 | -0.586 | 1.691 | 6456.1 | `doorkey_token_dense_fullobs` |

### keycorridors3r1

| variant | seed | greedy_eval | fullobs | eval_return | eval_success | final_train_return | best_train_return | train_eval_gap | explained_variance | entropy | throughput | run |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flat_dense | 7 | true | false | 0.000 | 0.000 | 0.337 | 0.337 | 0.337 | 0.559 | 1.726 | 9756.0 | `keycorridor_flat_dense` |
| single_expert | 7 | true | false | 0.000 | 0.000 | 0.000 | 0.687 | 0.000 | 0.075 | 1.576 | 7038.2 | `keycorridor_single_expert` |
| token_dense | 7 | true | false | 0.000 | 0.000 | 0.000 | 0.240 | 0.000 | -1.943 | 1.800 | 6533.3 | `keycorridor_token_dense` |

### memorys9

| variant | seed | greedy_eval | fullobs | eval_return | eval_success | final_train_return | best_train_return | train_eval_gap | explained_variance | entropy | throughput | run |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| token_dense | 7 | true | false | 0.000 | 0.000 | 0.422 | 0.561 | 0.422 | 0.013 | 1.255 | 6502.7 | `memory_token_dense` |
| token_gru | 7 | true | false | 0.000 | 0.000 | 0.445 | 0.537 | 0.445 | 0.188 | 1.592 | 6260.1 | `memory_token_gru` |
