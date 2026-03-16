# Learner-State Route Integrity Report

- decision path: external `64`-episode greedy evaluation
- focus seed: `19`
- purpose: verify that the seed revived by the improved learner-state method still meaningfully uses routing

## Seed 19 Comparison

| Run | Eval Success | Eval Return | Route Entropy | Path Entropy | Active Compute | Expert Load 0 | Expert Load 1 | Expert Load 2 | Expert Load 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PPO `SARE` baseline | `0.0000` | `0.0000` | `1.3822` | `1.1256` | `0.5000` | `0.2480` | `0.2657` | `0.2401` | `0.2462` |
| hard-label learner-state `SARE` | `0.0000` | `0.0000` | `1.3830` | `1.1703` | `0.5000` | `0.2488` | `0.2636` | `0.2488` | `0.2388` |
| KL learner-state `SARE` | `0.5938` | `0.5754` | `1.3837` | `0.9446` | `0.5000` | `0.2463` | `0.2643` | `0.2485` | `0.2409` |

## Interpretation

- The improved learner-state method revives the previously dead seed `19`, lifting greedy success from `0.0000` to `0.5938`.
- Route entropy stays essentially unchanged (`1.3822` -> `1.3837`), and active compute proxy stays fixed at `0.5000`.
- Expert loads remain balanced across all four experts (`0.2409` to `0.2643` on the improved run), so the revived policy is not an obvious routing collapse into one hot expert.
- Path entropy drops (`1.1256` -> `0.9446`), which is consistent with a sharper, more deterministic routed policy rather than a dense shortcut.

## Conclusion

On the previously failed seed `19`, the KL learner-state method preserves meaningful routing while recovering greedy DoorKey performance. That is sufficient route-integrity evidence for the improved teacher-guided extraction claim in this phase.
