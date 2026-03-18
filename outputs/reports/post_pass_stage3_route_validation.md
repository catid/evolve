# Post-PASS Stage 3 Route Validation

| Lane | Seed | Baseline | Fixed-Router Drop | Route-Randomization Drop | Worst Ablation Drop | Final Disagreement Δ | Final Post-Unlock Frac Δ | Best-Round Shift |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fresh | 29 | 1.0000 | 1.0000 | 0.0938 | 1.0000 | 0.0000 | 0.9507 | 0 |
| fresh_final | 47 | 0.4531 | 0.4531 | 0.4531 | 0.4531 | 0.0000 | 0.7646 | 3 |
| fresh_final | 59 | 0.4219 | 0.4219 | 0.4219 | 0.4219 | 0.0000 | 0.9795 | 0 |
| original | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | -0.9763 | 0.4201 | -1 |

## Interpretation

- stage-3 route status: `pass`
- The added strong cases use two additional historically strong recovered routed seeds, while the weak cases remain the historically mixed retry-block seeds `47` and `59`.
- The summary deltas show whether post_unlock_weighted shifts late-phase disagreement, route entropy, and cleanup-round timing relative to the frozen learner-state SARE baseline on the same seed.
