# Memory Actor-Hidden Partial-Shift Confirmation

- purpose: confirm the new `partial_shift25` actor-hidden FiLM result over a longer evaluation horizon after the initial 64-episode probe showed identical success across greedy and sampled modes
- task: `MiniGrid-MemoryS9-v0`
- compared lines: `por_base`, `por_actor_hidden_scale_film_mild`, `por_actor_hidden_partial_shift25`
- episodes per evaluation: `256`
- modes: greedy, `t=0.05`, `t=0.055`, `t=0.08`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Lower Margin | Gap Margin | Shoulder Margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3711` | `0.4414` | `0.4805` | `0.2990` | `0.3821` | `0.4303` | `0.6881` |
| `por_actor_hidden_scale_film_mild` | `0.0000` | `0.4062` | `0.4688` | `0.4727` | `0.2754` | `0.2891` | `0.2951` | `0.3244` |
| `por_actor_hidden_partial_shift25` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.6452` | `2.6452` | `2.6452` | `2.6452` |

## Deltas

| Variant | Delta Greedy vs Base | Delta Lower vs Base | Delta Gap vs Base | Delta Shoulder vs Base | Delta Greedy vs Mild | Delta Lower vs Mild | Delta Gap vs Mild | Delta Shoulder vs Mild |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift25` | `+0.4688` | `+0.0977` | `+0.0273` | `-0.0117` | `+0.4688` | `+0.0625` | `+0.0000` | `-0.0039` |

## Interpretation

- `partial_shift25` keeps nonzero greedy success over `256` episodes (`0.4688`) while base stays flat at `0.0000`.
- vs base, `partial_shift25` changes the sampled band by `+0.0977` / `+0.0273` / `-0.0117` at `t=0.05 / 0.055 / 0.08`.
- vs the mild scale-only reference, `partial_shift25` changes the sampled band by `+0.0625` / `+0.0000` / `-0.0039`.
- This confirmation pass is only meant to validate the architecture probe itself; it is not a benchmark-pack path and does not change the accepted active benchmark state.
