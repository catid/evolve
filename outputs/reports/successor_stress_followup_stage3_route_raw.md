# Causal Route-Dependence Report

- external evaluation episodes per probe: `64`
- git commit: `cfcc187c10038e10eb4a5358c77965803706fdd1`
- git dirty: `True`

| Lane | Seed | Probe | Detail | Greedy Success | Greedy Return | Route Entropy | Active Compute |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| prospective_f | 233 | baseline | - | 1.0000 | 0.9632 | 1.3759 | 0.5000 |
| prospective_f | 233 | expert_ablation | 0 | 0.0000 | 0.0000 | 1.3762 | 0.5000 |
| prospective_f | 233 | expert_ablation | 1 | 0.0000 | 0.0000 | 1.3763 | 0.5000 |
| prospective_f | 233 | expert_ablation | 2 | 0.0000 | 0.0000 | 1.3757 | 0.5000 |
| prospective_f | 233 | expert_ablation | 3 | 0.0000 | 0.0000 | 1.3756 | 0.5000 |
| prospective_f | 233 | router_override | most_used_pair | 0.0000 | 0.0000 | 0.6931 | 0.5000 |
| prospective_f | 233 | route_randomization | uniform_topk_random | 0.0000 | 0.0000 | 0.6931 | 0.5000 |

## Interpretation

- `prospective_f` seed `233` uses top experts `[2, 3]` for the fixed-router probe. Baseline greedy success is `1.0000`; fixed-router drop is `1.0000`, random-routing drop is `1.0000`, and worst single-expert ablation drop is `1.0000` (expert `0`).
- Learned routing assignments are causally relevant under this bounded probe family: disrupting routing materially degrades greedy DoorKey success.
