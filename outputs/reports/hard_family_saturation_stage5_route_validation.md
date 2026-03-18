# Hard-Family Saturation Stage 5 Route Validation

- best candidate: `round6`

| Case | Lane | Seed | Baseline | Fixed-Router Drop | Route-Randomization Drop | Worst Ablation Drop | Final Disagreement Δ | Final Post-Unlock Frac Δ | Final Route Entropy Δ | Final Path Entropy Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | post_pass_f | 137 | 1.0000 | 1.0000 | 0.9844 | 1.0000 | -0.0015 | -0.0037 | 0.0002 | 0.0499 |
| holdout | fresh_final | 47 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | -0.9747 | -0.3122 | -0.0075 | -0.0037 |
| healthy | post_pass_a | 67 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | -0.1209 | -0.3881 | 0.0017 | 0.1019 |

## Interpretation

- stage-5 status: `pass`
- The route summary compares the best hard-family candidate against the current thaw-qualified incumbent on the same dev, holdout, and healthy seeds so a hard-family gain cannot hide by erasing route structure.
