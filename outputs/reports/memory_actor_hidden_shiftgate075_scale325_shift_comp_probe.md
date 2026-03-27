# Memory ShiftGate075 Scale325 Adaptive-Floor2875 Shift-Compensation Probe

- hypothesis: `adaptive_floor2875` may be slightly under-shifted relative to the restored `shiftgate075_scale325` winner, so stronger shift compensation could recover the lost `0.0625` plateau without giving up stability
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `6`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Adaptive Scale Mean | Shift Compensation Mean | Comp Enabled | Comp Scale | Scale Floor | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.0000` | `0` | `0.00` | `0.0000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift22` | `0.0000` | `0.0000` | `0.0312` | `0.1406` | `0.3828` | `0.3837` | `0.4429` | `0.4369` | `0.4524` | `0.3719` | `0.1302` | `0.1302` | `0` | `0.00` | `0.0000` | `0.0281` | `0.0035` |
| `por_actor_hidden_partial_shift22_shiftgate075_scale325` | `0.5312` | `0.5312` | `0.5312` | `0.5312` | `2.1753` | `2.1753` | `2.1716` | `0.5579` | `0.5714` | `0.3295` | `0.1071` | `0.1412` | `0` | `0.00` | `0.0000` | `0.0500` | `0.0156` |
| `por_actor_hidden_partial_shift22_shiftgate075_scale325_adaptive_floor2875` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `3.6534` | `3.6534` | `3.6471` | `0.5401` | `0.5481` | `0.3352` | `0.1006` | `0.1430` | `0` | `0.00` | `0.2875` | `0.2450` | `0.0880` |
| `por_actor_hidden_partial_shift22_shiftgate075_scale325_adaptive_floor2875_shiftcomp100` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.9520` | `0.9520` | `0.9520` | `0.3324` | `0.3529` | `0.3785` | `0.1142` | `0.1681` | `1` | `1.00` | `0.2875` | `0.0235` | `0.0060` |
| `por_actor_hidden_partial_shift22_shiftgate075_scale325_adaptive_floor2875_shiftcomp200` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.6677` | `1.6677` | `1.6677` | `0.1473` | `0.2857` | `0.3882` | `0.1172` | `0.1824` | `1` | `2.00` | `0.2875` | `0.1096` | `0.0528` |

## Deltas vs Adaptive-Floor2875

- `shiftcomp100`: greedy -0.4688, lower -0.4688, gap -0.4688, shoulder -0.4688
- `shiftcomp200`: greedy -0.4688, lower -0.4688, gap -0.4688, shoulder -0.4688

## Conclusion

- stronger shift compensation does not improve on `adaptive_floor2875`; this mild scale-floor branch is not fixed by simply boosting the additive hidden shift
