# Causal Route-Dependence Report

- external evaluation episodes per probe: `64`
- git commit: `85e2aaadae7c741e625431dbf494372007eb001c`
- git dirty: `True`

| Lane | Seed | Probe | Detail | Greedy Success | Greedy Return | Route Entropy | Active Compute |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| prospective_h | 269 | baseline | - | 0.9844 | 0.9499 | 1.3624 | 0.5000 |
| prospective_h | 269 | expert_ablation | 0 | 0.0000 | 0.0000 | 1.3634 | 0.5000 |
| prospective_h | 269 | expert_ablation | 1 | 0.0000 | 0.0000 | 1.3632 | 0.5000 |
| prospective_h | 269 | expert_ablation | 2 | 0.0000 | 0.0000 | 1.3637 | 0.5000 |
| prospective_h | 269 | expert_ablation | 3 | 0.7500 | 0.7238 | 1.3629 | 0.5000 |
| prospective_h | 269 | router_override | most_used_pair | 0.0000 | 0.0000 | 0.6931 | 0.5000 |
| prospective_h | 269 | route_randomization | uniform_topk_random | 0.0000 | 0.0000 | 0.6931 | 0.5000 |

## Interpretation

- `prospective_h` seed `269` uses top experts `[1, 0]` for the fixed-router probe. Baseline greedy success is `0.9844`; fixed-router drop is `0.9844`, random-routing drop is `0.9844`, and worst single-expert ablation drop is `0.9844` (expert `0`).
- Learned routing assignments are causally relevant under this bounded probe family: disrupting routing materially degrades greedy DoorKey success.
