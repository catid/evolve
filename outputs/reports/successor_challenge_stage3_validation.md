# Successor Carryover Challenge Stage 3 Validation

- selected challenger: `round7`
- hard route status: `pass`
- healthy route status: `pass`
- stability status: `pass`

## Route Probes

| Case | Lane | Seed | Baseline | Fixed-Router Drop | Randomization Drop | Worst-Ablation Drop | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| hard | fresh_final | 53 | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `pass` |
| healthy | fresh | 23 | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `pass` |

## Stability

| Line | Case | Best Round | Best Greedy | Final Greedy | Stability |
| --- | --- | ---: | ---: | ---: | --- |
| `round6` | hard | `5` | `1.0000` | `1.0000` | `stable_plateau` |
| `round7` | hard | `5` | `1.0000` | `1.0000` | `stable_plateau` |
| `round6` | healthy | `4` | `1.0000` | `1.0000` | `stable_plateau` |
| `round7` | healthy | `4` | `1.0000` | `1.0000` | `stable_plateau` |
