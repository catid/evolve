# Run Summary

## Aggregate By Environment / Variant

### empty_5x5

| variant | seeds | eval_return | eval_success | best_train_return | train_eval_gap | throughput | compute_proxy | route_entropy | hop_count | rel_usage | option_duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat_dense | 1 | 0.955 | 1.000 | 0.955 | -0.000 | 9464.5 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| token_dense | 1 | 0.955 | 1.000 | 0.955 | -0.000 | 6451.4 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### fourrooms

| variant | seeds | eval_return | eval_success | best_train_return | train_eval_gap | throughput | compute_proxy | route_entropy | hop_count | rel_usage | option_duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat_dense | 1 | 0.000 | 0.000 | 0.220 | 0.084 | 9287.4 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| token_dense | 1 | 0.000 | 0.000 | 0.376 | 0.000 | 6340.2 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Per-Run Details

### empty_5x5

| variant | seed | greedy_eval | fullobs | eval_return | eval_success | final_train_return | best_train_return | train_eval_gap | explained_variance | entropy | throughput | run |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flat_dense | 7 | true | false | 0.955 | 1.000 | 0.955 | 0.955 | -0.000 | 0.727 | 0.018 | 9464.5 | `empty_flat_dense` |
| token_dense | 7 | true | false | 0.955 | 1.000 | 0.955 | 0.955 | -0.000 | 0.962 | 0.011 | 6451.4 | `empty_token_dense` |

### fourrooms

| variant | seed | greedy_eval | fullobs | eval_return | eval_success | final_train_return | best_train_return | train_eval_gap | explained_variance | entropy | throughput | run |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flat_dense | 7 | true | false | 0.000 | 0.000 | 0.084 | 0.220 | 0.084 | -0.642 | 1.731 | 9287.4 | `fourrooms_flat_dense` |
| token_dense | 7 | true | false | 0.000 | 0.000 | 0.000 | 0.376 | 0.000 | -2.094 | 1.691 | 6340.2 | `fourrooms_token_dense` |
