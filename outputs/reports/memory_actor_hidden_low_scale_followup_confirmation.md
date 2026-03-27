# Memory Actor-Hidden Low-Scale Follow-Up Confirmation

- follow-up to the 64-episode screen in `memory_actor_hidden_low_scale_followup_probe.md`
- task: `MiniGrid-MemoryS9-v0`
- evaluation episodes per mode: `256`
- matched rerun roots: base POR, incumbent `partial_shift225`, low-scale `scale325`, low-scale `scale30`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3711` | `0.4414` | `0.4805` | `0.2990` | `0.4303` | `0.6881` |
| `por_actor_hidden_partial_shift225` | `0.4688` | `0.4727` | `0.4727` | `0.4688` | `2.6143` | `2.5947` | `2.5555` |
| `por_actor_hidden_partial_shift225_scale325` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.0609` | `1.0609` | `1.0609` |
| `por_actor_hidden_partial_shift225_scale30` | `0.0000` | `0.5352` | `0.5273` | `0.5234` | `0.1892` | `0.3097` | `0.3959` |

## Deltas

- `scale30` versus incumbent `partial_shift225`: greedy `-0.4688`, lower `+0.0625`, gap `+0.0547`, shoulder `+0.0547`
- `scale30` versus base POR: greedy `+0.0000`, lower `+0.1641`, gap `+0.0859`, shoulder `+0.0430`
- `scale325` versus incumbent `partial_shift225`: greedy `-0.4688`, lower `-0.4727`, gap `-0.4727`, shoulder `-0.4688`

## Conclusion

- incumbent `partial_shift225` remains the only variant on this local surface with nonzero greedy success: `0.4688`
- lower scale `0.30` is a real sampled-band improver, not 64-episode noise: it reaches `0.5352 / 0.5273 / 0.5234` versus incumbent `0.4727 / 0.4727 / 0.4688`
- but `scale30` gives up the greedy conversion entirely at `0.0000`
- `scale325` is fully dead at `0.0000 / 0.0000 / 0.0000 / 0.0000`
- interpretation: lowering overall hidden-FiLM scale below `0.35` creates a real greedy-versus-sampled trade-off on the `partial_shift225` surface instead of a strictly better architecture point
