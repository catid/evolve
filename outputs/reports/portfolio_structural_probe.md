# Portfolio Structural Fallback Probe

- source campaign: `lss_portfolio_structural_probe`
- git commit: `ef793d814b4f3ed05bfc8973b617fecfe49b18ee`
- git dirty: `True`
- support status: `measured_support_regression`

## Seed Comparison

| Lane | Seed | round6 | round7 | door3_post5 | token_dense | single_expert |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `prospective_c` | 193 | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `prospective_f` | 233 | `1.0000` | `1.0000` | `0.4531` | `0.6250` | `1.0000` |

## Interpretation

- `prospective_c/193` remains a global-hard sentinel. If all compared lines are still `0.0000`, this probe does not reopen that seed as a near-term optimization target.
- `prospective_f/233` is the decisive measurement gap for the structural fallback. If `door3_post5` matches `round7` and `round6` there at `1.0000`, the structural fallback is now measured cleanly on the support seed instead of merely inferred from the weakness lane.
- Even if `door3_post5` measures cleanly on `233`, it still remains a structural fallback rather than the new conservative default unless it offers some advantage beyond what the cheaper `round7` prior already provides.
