# Causal Route-Dependence Report

- external evaluation episodes per probe: `64`
- git commit: `6e7277a8d8489cb74927ff7c7b3e072809491bab`
- git dirty: `True`

| Lane | Seed | Probe | Detail | Greedy Success | Greedy Return | Route Entropy | Active Compute |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| prospective_i | 281 | baseline | - | 1.0000 | 0.9650 | 1.3705 | 0.5000 |
| prospective_i | 281 | expert_ablation | 0 | 0.5156 | 0.4952 | 1.3705 | 0.5000 |
| prospective_i | 281 | expert_ablation | 1 | 0.0000 | 0.0000 | 1.3702 | 0.5000 |
| prospective_i | 281 | expert_ablation | 2 | 0.0000 | 0.0000 | 1.3703 | 0.5000 |
| prospective_i | 281 | expert_ablation | 3 | 0.0000 | 0.0000 | 1.3702 | 0.5000 |
| prospective_i | 281 | router_override | most_used_pair | 0.0000 | 0.0000 | 0.6931 | 0.5000 |
| prospective_i | 281 | route_randomization | uniform_topk_random | 0.0000 | 0.0000 | 0.6931 | 0.5000 |

## Interpretation

- `prospective_i` seed `281` uses top experts `[2, 3]` for the fixed-router probe. Baseline greedy success is `1.0000`; fixed-router drop is `1.0000`, random-routing drop is `1.0000`, and worst single-expert ablation drop is `1.0000` (expert `1`).
- Learned routing assignments are causally relevant under this bounded probe family: disrupting routing materially degrades greedy DoorKey success.
