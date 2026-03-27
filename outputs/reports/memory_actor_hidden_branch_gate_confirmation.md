# Memory Actor-Hidden Branch-Gate Confirmation

- purpose: determine whether the branch-specific shift-only gate mix at `0.90` is a real tie, a gain, or a regression relative to `partial_shift22`
- evaluation episodes per mode: `256`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift22` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.3520` | `2.3520` | `2.3520` |
| `por_actor_hidden_partial_shift22_shiftmix90` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.5331` | `2.5331` | `2.5331` |

## Deltas vs Partial-Shift22

- greedy: `+0.0000`
- lower t0.05: `+0.0000`
- gap t0.055: `+0.0000`
- shoulder t0.08: `+0.0000`

## Outcome

The 256-episode confirmation did not show a real gain over `partial_shift22`. The branch-specific shift-gate variant should be treated as a tie or regression, not a new best point.
