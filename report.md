# DoorKey Control-Decomposition Report

## Current Conclusion

- `flat_dense` remains the strongest verified greedy DoorKey control.
- The original tokenized gap was real, but it was not one thing:
  - `token_dense` was both underpowered and badly calibrated under greedy extraction.
  - `single_expert` and `SARE` were much closer to solvable policies, but their action selection stayed too soft for greedy evaluation.
- The smallest successful control-side recovery was not a new architecture. It was reducing PPO entropy pressure for `token_dense` from `0.01` to `0.001`.
- Under that recovered setting, `token_dense` becomes a competent greedy DoorKey control.
- A fair matched `SARE` rerun on the same setting still loses under greedy evaluation.

## What Was Audited And Fixed

- Reproduced the accepted DoorKey comparison under the current experiment lane and saved [reproduction_note.md](/home/catid/evolve/outputs/reports/reproduction_note.md).
- Fixed a provenance gap in `run_meta.json`: future runs now record both `git_commit` and `git_dirty`.
- Added rollout/eval action-selection diagnostics:
  - action entropy
  - max action probability
  - top-1 vs top-2 logit margin
  - greedy-match rate
- Added temperature-scaled sampled evaluation to separate policy quality from greedy extraction quality.
- Added lightweight token representation metrics:
  - pooled norm
  - token feature std
  - token pairwise cosine proxy

## Policy-Extraction Findings

Source artifact: [policy_extraction_report.md](/home/catid/evolve/outputs/reports/policy_extraction_report.md)

Original accepted DoorKey setting:

- `flat_dense`: greedy success `1.000`, eval max-prob `0.995`, logit margin `6.671`
- `token_dense`: greedy success `0.000`, best sampled success `0.125`, eval max-prob `0.315`, logit margin `0.330`
- `single_expert`: greedy success `0.000`, best sampled success `1.000`
- `SARE`: greedy success `0.000`, best sampled success `1.000`

Interpretation:

- `flat_dense` learns a sharp deterministic policy.
- `token_dense` is weak even when sampling is allowed, so its problem is not only greedy extraction.
- `single_expert` and `SARE` already contain competent sampled policies on DoorKey, but they do not extract a strong greedy policy. Their failure mode is much more calibration/extraction-heavy than `token_dense`.

## Tokenization-Gap Findings

Source artifact: [tokenization_gap_report.md](/home/catid/evolve/outputs/reports/tokenization_gap_report.md)

DoorKey tokenized diagnostics:

| Run | Greedy Success | Best Sampled Success | Train Return | Repr Std | Repr Cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline tokenized | `0.000` | `0.125` | `0.310` | `0.241` | `0.932` |
| fully observed | `0.000` | `0.688` | `0.267` | `0.229` | `0.942` |
| depth-4 | `0.000` | `0.125` | `0.088` | `0.151` | `0.970` |
| depth-4 + fully observed | `0.000` | `0.938` | `0.718` | `0.204` | `0.948` |

Interpretation:

- Full observation helps the tokenized path a lot under sampled evaluation, so partial observability is part of the gap.
- Depth alone does not fix the partial-observation tokenized control; in this implementation it actually looks more collapsed.
- Even with full observation and deeper token mixing, greedy extraction still fails. So the remaining problem is not just representation quality; it is also action calibration.

## Smallest Successful Recovery

Source artifact: [outputs/experiments/token_recovery/report.md](/home/catid/evolve/outputs/experiments/token_recovery/report.md)

Bounded sweep:

- `token_dense`, `ent_coef=0.001`: greedy success `0.750`, train return `0.942`, rollout entropy `0.695`, repr std `0.467`, repr cosine `0.735`
- `token_dense`, `ent_coef=0.0`: greedy success `0.000`
- `single_expert`, `ent_coef=0.001`: greedy success `0.000`, best sampled success `0.750`
- `single_expert`, `ent_coef=0.0`: greedy success `0.000`

Interpretation:

- The smallest successful change in this repo is `token_dense` with `ent_coef=0.001`.
- That change both sharpens action selection and materially improves representation health.
- Pushing entropy all the way to zero is too aggressive and does not recover the control.
- The same sweep does not recover `single_expert`, so there is not one universal calibration fix for the whole tokenized path.

## Fair SARE Retest

Source artifact: [outputs/experiments/sare_retest/report.md](/home/catid/evolve/outputs/experiments/sare_retest/report.md)

Matched DoorKey setting (`ent_coef=0.001`):

| Variant | Greedy Success | Best Sampled Success | Train Return | Throughput |
| --- | ---: | ---: | ---: | ---: |
| `flat_dense` | `1.000` | `1.000` | `0.960` | `9062.6` |
| `token_dense` | `0.750` | `1.000` | `0.942` | `6047.8` |
| `single_expert` | `0.000` | `0.750` | `0.291` | `6509.1` |
| `SARE` | `0.000` | `1.000` | `0.744` | `5748.6` |

Interpretation:

- `SARE` improves strongly under sampled evaluation, but it still fails under greedy evaluation on the same setting where `token_dense` now succeeds.
- So the honest routed conclusion is still negative for the repo’s main greedy DoorKey benchmark.
- The recovered tokenized control made the routed comparison fairer, and `SARE` still did not win that fair test.

## Recommendation

- Continue using `flat_dense` as the strongest verified DoorKey control.
- Treat `token_dense` with `ent_coef=0.001` as the current canonical recovered tokenized DoorKey control.
- Pause further routed-architecture work in this repo for greedy-eval claims.
- If work continues, the highest-value follow-up is not new routing variants. It is policy extraction / calibration for routed token policies that already look competent under sampled evaluation.
