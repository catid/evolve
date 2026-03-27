# Memory Actor-Hidden Powered Adaptive Scale-Floor Probe

- hypothesis: a concave adaptive floor curve may keep higher-duration states closer to the incumbent `partial_shift22` scale while still softening lower-duration states enough to preserve the sampled-band gain
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `5`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Floor Power | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Adaptive Scale Mean | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.00` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.0000` |
| `partial_shift22` | `1.00` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.3544` | `2.3544` | `2.3544` | `0.6509` | `0.6625` | `0.1433` | `0.1073` | `0.0228` |
| `adaptive_floor25` | `1.00` | `0.3125` | `0.4688` | `0.4844` | `0.4688` | `1.0770` | `1.4543` | `1.5349` | `0.4404` | `0.4500` | `0.1141` | `0.0285` | `0.0078` |
| `adaptive_floor25_power075` | `0.75` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.7558` | `0.7558` | `0.7558` | `0.5178` | `0.5366` | `0.1107` | `0.0236` | `0.0065` |
| `adaptive_floor25_power05` | `0.50` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.4725` | `1.4725` | `1.4725` | `0.4933` | `0.5135` | `0.2343` | `0.3526` | `0.0824` |

## Deltas vs Adaptive-Floor25

- `adaptive_floor25_power075`: greedy `-0.3125`, lower `-0.4688`, gap `-0.4844`, shoulder `-0.4688`
- `adaptive_floor25_power05`: greedy `-0.3125`, lower `-0.4688`, gap `-0.4844`, shoulder `-0.4688`

## Conclusion

- no powered floor variant improves the adaptive-floor trade-off enough to justify a confirmation pass
