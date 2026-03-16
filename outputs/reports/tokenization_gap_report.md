# Tokenization Gap Report

## Question

What explains the weak original `token_dense` DoorKey result, and what is the smallest change that makes a tokenized control competitive enough for a fair routed comparison?

Relevant artifact roots:

- [outputs/diagnostics/tokenization_gap/report.md](../diagnostics/tokenization_gap/report.md)
- [outputs/experiments/token_recovery/report.md](../experiments/token_recovery/report.md)
- [outputs/experiments/sare_retest/report.md](../experiments/sare_retest/report.md)

## Phase 1: Tokenization-Gap Diagnostics

| Run | Greedy Success | Best Sampled Success | Train Return | Repr Std | Repr Cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline tokenized | `0.000` | `0.125` | `0.310` | `0.241` | `0.932` |
| fully observed | `0.000` | `0.688` | `0.267` | `0.229` | `0.942` |
| depth-4 | `0.000` | `0.125` | `0.088` | `0.151` | `0.970` |
| depth-4 + fully observed | `0.000` | `0.938` | `0.718` | `0.204` | `0.948` |

## What Those Diagnostics Mean

- Partial observability matters. Moving to fully observed DoorKey raises sampled tokenized success from `0.125` to `0.688` even at the original depth.
- Deeper token mixing alone does not rescue partial-observation `token_dense`. In this implementation it looks more collapsed, not less.
- Full observation plus depth helps a lot under sampled evaluation, but greedy success is still `0.000`. So the remaining gap is not just representation quality; action extraction is also a distinct problem.

## Phase 2: Smallest Successful Intervention

Bounded entropy sweep on controls only:

| Run | Greedy Success | Best Sampled Success | Train Return | Rollout Entropy | Repr Std | Repr Cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `token_dense`, `ent_coef=0.001` | `0.750` | `1.000` | `0.942` | `0.695` | `0.467` | `0.735` |
| `token_dense`, `ent_coef=0.0` | `0.000` | `0.344` | `0.141` | `1.544` | `0.219` | `0.943` |
| `single_expert`, `ent_coef=0.001` | `0.000` | `0.750` | `0.291` | `1.397` | `0.630` | `0.521` |
| `single_expert`, `ent_coef=0.0` | `0.000` | `0.031` | `0.103` | `1.670` | `0.648` | `0.487` |

## Smallest Working Change

The smallest successful recovery in this repo is:

- keep the existing `token_dense` architecture
- keep the existing DoorKey setup
- lower PPO `ent_coef` from `0.01` to `0.001`

That change does two useful things at once:

- it sharpens the policy enough to reach greedy DoorKey success `0.750`
- it improves token representation health, with much higher token feature variance and much less pairwise collapse

Driving entropy all the way to zero does not help; it destabilizes the tokenized control instead of recovering it.

## Phase 3: Fair SARE Retest

Matched setting: `ent_coef=0.001`

| Variant | Greedy Success | Best Sampled Success | Train Return |
| --- | ---: | ---: | ---: |
| `flat_dense` | `1.000` | `1.000` | `0.960` |
| recovered `token_dense` | `0.750` | `1.000` | `0.942` |
| `single_expert` | `0.000` | `0.750` | `0.291` |
| `SARE` | `0.000` | `1.000` | `0.744` |

## Final Answer

The flat-dense vs tokenized gap came from two different problems:

1. The original `token_dense` control was genuinely weak under partial observation, not just badly evaluated.
2. A separate extraction/calibration problem remained even when the tokenized policy became strong under sampled evaluation.

The smallest change that produced a fairer tokenized control was `token_dense` with `ent_coef=0.001`.

That recovered control is good enough to justify a fair routed comparison, and the matched `SARE` rerun still loses under greedy DoorKey evaluation.

## Recommendation

- Keep `token_dense` with `ent_coef=0.001` as the canonical recovered tokenized DoorKey control.
- Do not claim a routed win on DoorKey in this repo.
- Pause further routed-architecture work for greedy-eval claims.
- If the project continues, focus on policy extraction/calibration for routed token policies rather than adding more routed variants.
