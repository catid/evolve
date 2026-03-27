# Memory Next Candidate Canonicalization

## Goal

Collapse the seven surviving `Memory` aliases from the large Memory-next program to one canonical task-scoped candidate path.

## Surviving Alias Cluster

All seven surviving aliases from Stage D2/D3 held the same benchmark-facing metrics:

| candidate_id | family | greedy | lower | gap | shoulder | healthy_gap | stability_greedy | compute_cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `partial225_gap_t055` | actor_hidden_continuation | `0.5208` | `0.5208` | `0.5208` | `0.5208` | `0.4219` | `0.5469` | `1.0` |
| `partial225_greedy` | actor_hidden_continuation | `0.5208` | `0.5208` | `0.5208` | `0.5208` | `0.4219` | `0.5469` | `1.0` |
| `select_partial225_greedy` | checkpoint_selection | `0.5208` | `0.5208` | `0.5208` | `0.5208` | `0.4219` | `0.5469` | `1.0` |
| `partial22_conf_gate085_t055` | sampled_to_greedy_decode | `0.5208` | `0.5208` | `0.5208` | `0.5208` | `0.4219` | `0.5469` | `1.0` |
| `partial22_conf_gate092_t055` | sampled_to_greedy_decode | `0.5208` | `0.5208` | `0.5208` | `0.5208` | `0.4219` | `0.5469` | `1.0` |
| `partial22_cons2_t05` | small_branch_consensus | `0.5208` | `0.5208` | `0.5208` | `0.5208` | `0.4219` | `0.5469` | `2.0` |
| `partial22_cons2_t055` | small_branch_consensus | `0.5208` | `0.5208` | `0.5208` | `0.5208` | `0.4219` | `0.5469` | `2.0` |

Source reports:

- `outputs/reports/memory_next_stageB1_exploit_screening.md`
- `outputs/reports/memory_next_stageD1_verification.md`
- `outputs/reports/memory_next_stageD2_fairness_holdout.md`
- `outputs/reports/memory_next_stageD3_antiregression_stability.md`

## Canonical Choice

Canonical task-scoped Memory candidate: `partial225_greedy`

## Rationale

- It matches the best observed greedy, lower-band, gap-band, shoulder, holdout, healthy, and stability metrics.
- It keeps unit compute cost, unlike the consensus aliases.
- It is a direct greedy path, not a temperature-tagged wrapper like `partial225_gap_t055`.
- It avoids extra checkpoint-selection or confidence-threshold machinery while preserving the same measured result.
- It stays on the stronger `partial_shift225` checkpoint lineage that survived holdout and stability as a plain greedy candidate.

## Package Update

`outputs/reports/memory_next_candidate_pack.json` now uses `partial225_greedy` as the canonical packaged Memory candidate.
