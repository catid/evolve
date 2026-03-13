# Current Summary

- Overfit on `Empty-5x5` is solved by both `flat_dense` and `token_dense`.
- Sanity-tier `Empty-5x5` is solved by both controls under the normal multi-env PPO path.
- `FourRooms` shows weak nonzero greedy eval for `flat_dense` and remains zero for `token_dense` at the current budget.
- Fully observed `DoorKey` is solved by tuned `token_dense` under greedy eval.
- Standard `DoorKey` shows a major evaluation-mode mismatch: tuned `token_dense` is `0.000` under greedy eval but `1.000` success under sampled eval.
- `Memory` also shows evaluation-mode sensitivity: both `token_gru` and matched `token_dense` are `0.000` under greedy eval but nonzero under sampled eval at 60 updates; `token_gru` remains diagnostic-only until PPO sequence batching is fixed.
- Conclusion: controls are finally learning, but the repo still needs explicit policy-extraction diagnostics before routed comparisons are fair.
