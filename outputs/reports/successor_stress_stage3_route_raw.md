# Causal Route-Dependence Report

- external evaluation episodes per probe: `64`
- git commit: `08252b4af2213d2109fb923a74103bdf61d6e614`
- git dirty: `True`

| Lane | Seed | Probe | Detail | Greedy Success | Greedy Return | Route Entropy | Active Compute |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| prospective_d | 197 | baseline | - | 1.0000 | 0.9638 | 1.3755 | 0.5000 |
| prospective_d | 197 | expert_ablation | 0 | 0.2969 | 0.2878 | 1.3755 | 0.5000 |
| prospective_d | 197 | expert_ablation | 1 | 0.0000 | 0.0000 | 1.3752 | 0.5000 |
| prospective_d | 197 | expert_ablation | 2 | 0.0000 | 0.0000 | 1.3751 | 0.5000 |
| prospective_d | 197 | expert_ablation | 3 | 0.0000 | 0.0000 | 1.3757 | 0.5000 |
| prospective_d | 197 | router_override | most_used_pair | 0.0000 | 0.0000 | 0.6931 | 0.5000 |
| prospective_d | 197 | route_randomization | uniform_topk_random | 0.0000 | 0.0000 | 0.6931 | 0.5000 |

## Interpretation

- `prospective_d` seed `197` uses top experts `[2, 1]` for the fixed-router probe. Baseline greedy success is `1.0000`; fixed-router drop is `1.0000`, random-routing drop is `1.0000`, and worst single-expert ablation drop is `1.0000` (expert `1`).
- Learned routing assignments are causally relevant under this bounded probe family: disrupting routing materially degrades greedy DoorKey success.
