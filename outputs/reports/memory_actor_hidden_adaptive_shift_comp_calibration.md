# Memory Actor-Hidden Adaptive Shift-Compensation Calibration

- hypothesis: there may be an interior compensation point between `0.0` and `0.5` that improves the sampled band over `adaptive_floor25` while keeping some greedy success
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed calibration isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Comp Scale | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Adaptive Scale Mean | Shift Comp Mean | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptive_floor25` | `0.00` | `0.3125` | `0.4688` | `0.4844` | `0.4688` | `1.0770` | `1.4543` | `1.5349` | `0.4404` | `0.4500` | `0.1141` | `0.1380` | `0.0285` | `0.0078` |
| `adaptive_floor25_shiftcomp20` | `0.20` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.4984` | `1.4984` | `1.4984` | `0.2299` | `0.3333` | `0.0994` | `0.1267` | `0.0290` | `0.0060` |
| `adaptive_floor25_shiftcomp35` | `0.35` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.2628` | `0.2628` | `0.2628` | `0.3578` | `0.3750` | `0.0950` | `0.1251` | `0.0273` | `0.0081` |
| `adaptive_floor25_shiftcomp50` | `0.50` | `0.0000` | `0.5312` | `0.5312` | `0.5469` | `0.0295` | `1.5246` | `1.5603` | `0.5378` | `0.5532` | `0.1043` | `0.1390` | `0.0207` | `0.0041` |

## Deltas vs Adaptive-Floor25

- `adaptive_floor25_shiftcomp20`: greedy `-0.3125`, lower `-0.4688`, gap `-0.4844`, shoulder `-0.4688`
- `adaptive_floor25_shiftcomp35`: greedy `-0.3125`, lower `-0.4688`, gap `-0.4844`, shoulder `-0.4688`
- `adaptive_floor25_shiftcomp50`: greedy `-0.3125`, lower `+0.0625`, gap `+0.0469`, shoulder `+0.0781`

## Conclusion

- no interior compensation point improves the adaptive-floor trade-off enough to justify a confirmation pass
