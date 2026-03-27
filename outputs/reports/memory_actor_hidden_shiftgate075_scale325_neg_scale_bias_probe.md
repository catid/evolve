# Memory ShiftGate075 Scale325 Negative Scale-Bias Probe

- hypothesis: the current winner may benefit from selectively suppressing the multiplicative scale gate on a few over-active states, since the best fixed retune already moved scale downward from `0.35` to `0.325`
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `5`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Scale-Bias Scale | Gate Bias Mean | Scale Gate Bias Mean | Gate Mean | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift22` | `0.0000` | `0.0000` | `0.0312` | `0.1406` | `0.3828` | `0.3837` | `0.4429` | `0.4369` | `0.4524` | `0.0000` | `0.0000` | `0.0000` | `0.1302` | `0.0281` | `0.0035` |
| `por_actor_hidden_partial_shift22_shiftgate075_scale325` | `0.5312` | `0.5312` | `0.5312` | `0.5312` | `2.1753` | `2.1753` | `2.1716` | `0.5579` | `0.5714` | `0.0000` | `0.0000` | `0.0000` | `0.1241` | `0.0500` | `0.0156` |
| `por_actor_hidden_partial_shift22_shiftgate075_scale325_scalebiasneg005` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.9283` | `0.9283` | `0.9283` | `0.2922` | `0.5000` | `-0.0050` | `0.0045` | `-0.0000` | `0.3181` | `0.2230` | `0.0516` |
| `por_actor_hidden_partial_shift22_shiftgate075_scale325_scalebiasneg01` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.7141` | `0.7141` | `0.7141` | `0.5101` | `0.5263` | `-0.0100` | `-0.0011` | `0.0000` | `0.1143` | `0.0543` | `0.0192` |

## Deltas vs ShiftGate075 Scale325

- `scalebiasneg005`: greedy -0.5312, lower -0.5312, gap -0.5312, shoulder -0.5312
- `scalebiasneg01`: greedy -0.5312, lower -0.5312, gap -0.5312, shoulder -0.5312

## Conclusion

- signed negative scale bias does not improve on `shiftgate075_scale325`; the current winner does not want learned suppression on the multiplicative scale gate
