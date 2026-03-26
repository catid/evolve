# Memory Actor-Hidden Lower-Shift Confirmation

- purpose: verify whether the only live lower-shift newcomer (`partial_shift20`) survives a longer evaluation horizon and actually challenges `partial_shift25` on the Memory conversion branch
- task: `MiniGrid-MemoryS9-v0`
- compared lines: `por_base`, `por_actor_hidden_partial_shift20`, `por_actor_hidden_partial_shift25`
- episodes per evaluation: `256`
- modes: greedy, `t=0.05`, `t=0.055`, `t=0.08`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Lower Margin | Gap Margin | Shoulder Margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3711` | `0.4414` | `0.4805` | `0.2990` | `0.3821` | `0.4303` | `0.6881` |
| `por_actor_hidden_partial_shift20` | `0.0000` | `0.5000` | `0.4883` | `0.4766` | `0.2251` | `0.1640` | `0.1704` | `0.1674` |
| `por_actor_hidden_partial_shift25` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.6452` | `2.6452` | `2.6452` | `2.6452` |

## Deltas vs Partial-Shift25

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift20` | `-0.4688` | `+0.0312` | `+0.0195` | `+0.0078` |

## Interpretation

- `partial_shift20` greedy stays `0.0000` over `256` episodes, so it does not preserve the new greedy conversion behavior.
- vs `partial_shift25`, `partial_shift20` changes the sampled band by `+0.0312` / `+0.0195` / `+0.0078` at `t=0.05 / 0.055 / 0.08`.
- This confirms the lower-shift sweep as a negative follow-up: smaller shift weights do not beat the existing `partial_shift25` architecture result.
