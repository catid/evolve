# Causal Route-Dependence Report

- external evaluation episodes per probe: `64`
- git commit: `6e7277a8d8489cb74927ff7c7b3e072809491bab`
- git dirty: `True`

| Lane | Seed | Probe | Detail | Greedy Success | Greedy Return | Route Entropy | Active Compute |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| fresh | 23 | baseline | - | 1.0000 | 0.9641 | 1.3805 | 0.5000 |
| fresh | 23 | expert_ablation | 0 | 0.0000 | 0.0000 | 1.3806 | 0.5000 |
| fresh | 23 | expert_ablation | 1 | 0.0000 | 0.0000 | 1.3806 | 0.5000 |
| fresh | 23 | expert_ablation | 2 | 0.0000 | 0.0000 | 1.3808 | 0.5000 |
| fresh | 23 | expert_ablation | 3 | 0.0000 | 0.0000 | 1.3807 | 0.5000 |
| fresh | 23 | router_override | most_used_pair | 0.0000 | 0.0000 | 0.6931 | 0.5000 |
| fresh | 23 | route_randomization | uniform_topk_random | 0.0000 | 0.0000 | 0.6931 | 0.5000 |

## Interpretation

- `fresh` seed `23` uses top experts `[3, 1]` for the fixed-router probe. Baseline greedy success is `1.0000`; fixed-router drop is `1.0000`, random-routing drop is `1.0000`, and worst single-expert ablation drop is `1.0000` (expert `0`).
- Learned routing assignments are causally relevant under this bounded probe family: disrupting routing materially degrades greedy DoorKey success.
