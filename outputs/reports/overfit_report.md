# Run Summary

## Aggregate By Environment / Variant

### empty_5x5_fullobs

| variant | seeds | eval_return | eval_success | best_train_return | train_eval_gap | throughput | compute_proxy | route_entropy | hop_count | rel_usage | option_duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat_dense | 1 | 0.955 | 1.000 | 0.955 | 0.000 | 1074.8 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| token_dense | 1 | 0.955 | 1.000 | 0.955 | 0.000 | 269.7 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Per-Run Details

### empty_5x5_fullobs

| variant | seed | greedy_eval | fullobs | eval_return | eval_success | final_train_return | best_train_return | train_eval_gap | explained_variance | entropy | throughput | run |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flat_dense | 7 | true | true | 0.955 | 1.000 | 0.955 | 0.955 | 0.000 | 0.988 | 0.015 | 1074.8 | `overfit_empty5_flat_dense` |
| token_dense | 7 | true | true | 0.955 | 1.000 | 0.955 | 0.955 | 0.000 | 0.994 | 0.020 | 269.7 | `overfit_empty5_token_dense` |
