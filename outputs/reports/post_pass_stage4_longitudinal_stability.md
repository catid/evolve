# Post-PASS Stage 4 Longitudinal Stability

| Case | Lane | Seed | Source | Round Successes | Classification |
| --- | --- | --- | --- | --- | --- |
| weak | fresh_final | 53 | candidate | `0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0625, 0.0000, 0.0000, 0.5156, 0.5000, 0.5000, 0.5312, 0.5156, 0.5312, 0.5156, 0.5156` | `stable_plateau` |
| weak | fresh_final | 53 | baseline | `0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0625, 0.0000, 0.0000, 0.5156, 0.5000, 0.5000, 0.5312, 0.5156, 0.6406, 0.5156, 0.5156` | `stable_plateau` |
| strong | fresh | 23 | candidate | `0.0000, 0.1094, 0.0156, 0.0000, 0.0000, 0.2344, 0.0625, 0.0312, 0.5938, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000` | `stable_plateau` |
| strong | fresh | 23 | baseline | `0.0000, 0.1094, 0.0156, 0.0000, 0.0000, 0.2344, 0.0625, 0.0312, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000` | `stable_plateau` |

## Interpretation

- stage-4 stability status: `pass`
- The weak representative tracks the historically difficult retry block, while the stronger representative comes from the healthiest historical routed seed retained by the candidate.
- Candidate stability is treated as suspect if it reduces to a narrow one-checkpoint spike even when the gate metrics still clear.
