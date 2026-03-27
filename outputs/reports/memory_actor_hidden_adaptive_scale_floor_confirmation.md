# Memory Actor-Hidden Adaptive Scale-Floor Confirmation

- follow-up to the 64-episode screen in `memory_actor_hidden_adaptive_scale_floor_probe.md`
- task: `MiniGrid-MemoryS9-v0`
- evaluation episodes per mode: `256`
- matched rerun roots: base POR, incumbent `partial_shift22`, adaptive-floor survivor `adaptive_floor25`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Adaptive Scale Mean | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3711` | `0.4414` | `0.4805` | `0.2990` | `0.4303` | `0.6881` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift22` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.3520` | `2.3520` | `2.3520` | `0.6509` | `0.6625` | `0.4094` | `0.1433` | `0.1073` | `0.0228` |
| `por_actor_hidden_partial_shift22_adaptive_floor25` | `0.2617` | `0.4727` | `0.4727` | `0.4766` | `1.1006` | `1.4371` | `1.5074` | `0.4404` | `0.4500` | `0.3941` | `0.1141` | `0.0285` | `0.0078` |

## Deltas vs Partial-Shift22

- `adaptive_floor25`: greedy `-0.2070`, lower `+0.0039`, gap `+0.0039`, shoulder `+0.0078`

## Conclusion

- `adaptive_floor25` remains alive at `greedy/lower/gap/shoulder = 0.2617 / 0.4727 / 0.4727 / 0.4766`
- interpretation: duration-conditioned scale softening creates a real greedy-versus-sampled trade-off that survives beyond the 64-episode screen
