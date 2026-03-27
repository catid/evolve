# Memory Actor-Hidden Threshold Confirmation

- follow-up to the 64-episode screen in `memory_actor_hidden_threshold_probe.md`
- task: `MiniGrid-MemoryS9-v0`
- evaluation episodes per mode: `256`
- matched rerun roots: base POR, incumbent `partial_shift225`, threshold midpoints `partial_shift215` and `partial_shift22`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3711` | `0.4414` | `0.4805` | `0.2990` | `0.4303` | `0.6881` |
| `por_actor_hidden_partial_shift215` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.9839` | `1.9839` | `1.9839` |
| `por_actor_hidden_partial_shift22` | `0.4688` | `0.4766` | `0.4766` | `0.4688` | `1.3575` | `1.2885` | `1.2812` |
| `por_actor_hidden_partial_shift225` | `0.4688` | `0.4727` | `0.4727` | `0.4688` | `2.6143` | `2.5947` | `2.5555` |

## Deltas vs Partial-Shift225

- `partial_shift22`: greedy `+0.0000`, lower `+0.0039`, gap `+0.0039`, shoulder `+0.0000`
- `partial_shift215`: greedy `-0.4688`, lower `-0.4727`, gap `-0.4727`, shoulder `-0.4688`

## Conclusion

- `partial_shift22` stays fully tied with the incumbent at `0.4688 / 0.4766 / 0.4766 / 0.4688` on greedy/lower/gap/shoulder
- `partial_shift225` remains `0.4688 / 0.4727 / 0.4727 / 0.4688` at the same horizon
- `partial_shift215` is train-alive but eval-dead at `0.0000 / 0.0000 / 0.0000 / 0.0000`
- interpretation: the greedy-conversion threshold turns on by `0.22`, but this confirmation does not show a strict improvement over `partial_shift225`; it tightens the live plateau downward rather than moving the best-known point upward
