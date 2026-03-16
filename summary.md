# Current Summary

- `flat_dense` is still the best verified greedy DoorKey control.
- `token_dense` with `ppo.ent_coef=0.001` remains the canonical recovered tokenized DoorKey control.
- `single_expert` and `SARE` remain sampled-competent but greedy-failing on DoorKey.
- The bounded greedy-recovery campaign is now complete:
  - checkpoint dynamics found no missed good greedy checkpoint
  - entropy schedules found no nonzero greedy recovery
  - self-imitation from successful sampled trajectories found no nonzero greedy recovery
- Current recommendation: stop routed work for greedy-policy claims on DoorKey in this repo unless a new extraction method is justified as a separate project.
