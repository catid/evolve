# Causal Route-Dependence Report

- external evaluation episodes per probe: `64`
- git commit: `909d1d72796e0d6990d936db38a6a693ea0858fd`
- git dirty: `True`

| Lane | Seed | Probe | Detail | Greedy Success | Greedy Return | Route Entropy | Active Compute |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| prospective_j | 307 | baseline | - | 1.0000 | 0.9612 | 1.3839 | 0.5000 |
| prospective_j | 307 | expert_ablation | 0 | 0.0000 | 0.0000 | 1.3841 | 0.5000 |
| prospective_j | 307 | expert_ablation | 1 | 0.0000 | 0.0000 | 1.3844 | 0.5000 |
| prospective_j | 307 | expert_ablation | 2 | 0.0000 | 0.0000 | 1.3837 | 0.5000 |
| prospective_j | 307 | expert_ablation | 3 | 0.0000 | 0.0000 | 1.3841 | 0.5000 |
| prospective_j | 307 | router_override | most_used_pair | 0.0000 | 0.0000 | 0.6931 | 0.5000 |
| prospective_j | 307 | route_randomization | uniform_topk_random | 0.1250 | 0.0617 | 0.6931 | 0.5000 |

## Interpretation

- `prospective_j` seed `307` uses top experts `[1, 2]` for the fixed-router probe. Baseline greedy success is `1.0000`; fixed-router drop is `1.0000`, random-routing drop is `0.8750`, and worst single-expert ablation drop is `1.0000` (expert `0`).
- Learned routing assignments are causally relevant under this bounded probe family: disrupting routing materially degrades greedy DoorKey success.
