# Current Summary

- `flat_dense` is still the best verified greedy DoorKey control.
- The original tokenized gap split into two failure modes:
  - `token_dense` was weak and underconfident even when sampled.
  - `single_expert` and `SARE` were much better under sampled evaluation than under greedy evaluation.
- The smallest successful recovery was `token_dense` with `ppo.ent_coef=0.001`.
  - greedy DoorKey success: `0.750`
  - sampled DoorKey success: `1.000`
- A fair matched `SARE` rerun on that recovered setting still has greedy DoorKey success `0.000`, even though sampled success is `1.000`.
- Current recommendation: pause routed work for greedy-eval claims and treat policy extraction/calibration, not new architecture count, as the next bottleneck.
