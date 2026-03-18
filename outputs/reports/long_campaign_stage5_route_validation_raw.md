# Causal Route-Dependence Report

- external evaluation episodes per probe: `64`
- git commit: `e9f3257e660833253ed59f4c59923f5184139347`
- git dirty: `True`

| Lane | Seed | Probe | Detail | Greedy Success | Greedy Return | Route Entropy | Active Compute |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| fresh_final | 53 | baseline | - | 0.5156 | 0.4985 | 1.3756 | 0.5000 |
| fresh_final | 53 | expert_ablation | 0 | 0.0000 | 0.0000 | 1.3790 | 0.5000 |
| fresh_final | 53 | expert_ablation | 1 | 0.0000 | 0.0000 | 1.3796 | 0.5000 |
| fresh_final | 53 | expert_ablation | 2 | 0.5156 | 0.4985 | 1.3785 | 0.5000 |
| fresh_final | 53 | expert_ablation | 3 | 0.0000 | 0.0000 | 1.3789 | 0.5000 |
| fresh_final | 53 | router_override | most_used_pair | 0.0000 | 0.0000 | 0.6931 | 0.5000 |
| fresh_final | 53 | route_randomization | uniform_topk_random | 0.0000 | 0.0000 | 0.6931 | 0.5000 |
| fresh | 23 | baseline | - | 1.0000 | 0.9626 | 1.3813 | 0.5000 |
| fresh | 23 | expert_ablation | 0 | 0.0000 | 0.0000 | 1.3814 | 0.5000 |
| fresh | 23 | expert_ablation | 1 | 0.0000 | 0.0000 | 1.3814 | 0.5000 |
| fresh | 23 | expert_ablation | 2 | 0.0000 | 0.0000 | 1.3817 | 0.5000 |
| fresh | 23 | expert_ablation | 3 | 0.0000 | 0.0000 | 1.3814 | 0.5000 |
| fresh | 23 | router_override | most_used_pair | 0.0000 | 0.0000 | 0.6931 | 0.5000 |
| fresh | 23 | route_randomization | uniform_topk_random | 0.0000 | 0.0000 | 0.6931 | 0.5000 |

## Interpretation

- `fresh` seed `23` uses top experts `[3, 1]` for the fixed-router probe. Baseline greedy success is `1.0000`; fixed-router drop is `1.0000`, random-routing drop is `1.0000`, and worst single-expert ablation drop is `1.0000` (expert `0`).
- `fresh_final` seed `53` uses top experts `[3, 1]` for the fixed-router probe. Baseline greedy success is `0.5156`; fixed-router drop is `0.5156`, random-routing drop is `0.5156`, and worst single-expert ablation drop is `0.5156` (expert `0`).
- Learned routing assignments are causally relevant under this bounded probe family: disrupting routing materially degrades greedy DoorKey success.
