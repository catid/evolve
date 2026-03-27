# Memory ShiftGate075 Scale325 Negative Shift-Bias Probe

- hypothesis: the current winner may benefit from selectively suppressing the additive shift gate on a few over-active states, so signed negative shift-gate bias could improve the local plateau where positive bias failed
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `5`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Shift-Bias Scale | Gate Bias Mean | Shift Gate Bias Mean | Gate Mean | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift22` | `0.0000` | `0.0000` | `0.0312` | `0.1406` | `0.3828` | `0.3837` | `0.4429` | `0.4369` | `0.4524` | `0.0000` | `0.0000` | `0.0000` | `0.1302` | `0.0281` | `0.0035` |
| `por_actor_hidden_partial_shift22_shiftgate075_scale325` | `0.5312` | `0.5312` | `0.5312` | `0.5312` | `2.1753` | `2.1753` | `2.1716` | `0.5579` | `0.5714` | `0.0000` | `0.0000` | `0.0000` | `0.1241` | `0.0500` | `0.0156` |
| `por_actor_hidden_partial_shift22_shiftgate075_scale325_shiftbiasneg01` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.2645` | `2.2645` | `2.2645` | `0.5138` | `0.5270` | `-0.0100` | `-0.0061` | `0.0001` | `0.1418` | `0.1016` | `0.0309` |
| `por_actor_hidden_partial_shift22_shiftgate075_scale325_shiftbiasneg02` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.2752` | `0.2763` | `0.2765` | `0.0000` | `0.0000` | `-0.0200` | `0.0290` | `-0.0006` | `0.3223` | `0.1134` | `0.0209` |

## Deltas vs ShiftGate075 Scale325

- `shiftbiasneg01`: greedy -0.0625, lower -0.0625, gap -0.0625, shoulder -0.0625
- `shiftbiasneg02`: greedy -0.5312, lower -0.5312, gap -0.5312, shoulder -0.5312

## Conclusion

- signed negative shift bias does not improve on `shiftgate075_scale325`; the current winner does not want learned suppression on the additive shift gate
