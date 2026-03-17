# Causal Route-Dependence Report

- external evaluation episodes per probe: `64`
- git commit: `5b46a9e0545844ceefead4f559fc61c8e8fefcbf`
- git dirty: `True`

| Lane | Seed | Probe | Detail | Greedy Success | Greedy Return | Route Entropy | Active Compute |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| original | 7 | baseline | - | 1.0000 | 0.9607 | 1.3845 | 0.5000 |
| original | 7 | expert_ablation | 0 | 0.0000 | 0.0000 | 1.3845 | 0.5000 |
| original | 7 | expert_ablation | 1 | 0.0000 | 0.0000 | 1.3845 | 0.5000 |
| original | 7 | expert_ablation | 2 | 0.0000 | 0.0000 | 1.3845 | 0.5000 |
| original | 7 | expert_ablation | 3 | 0.0000 | 0.0000 | 1.3844 | 0.5000 |
| original | 7 | router_override | most_used_pair | 0.0000 | 0.0000 | 0.6931 | 0.5000 |
| original | 7 | route_randomization | uniform_topk_random | 0.0000 | 0.0000 | 0.6931 | 0.5000 |
| original | 19 | baseline | - | 0.5781 | 0.5596 | 1.3837 | 0.5000 |
| original | 19 | expert_ablation | 0 | 0.0000 | 0.0000 | 1.3839 | 0.5000 |
| original | 19 | expert_ablation | 1 | 0.0000 | 0.0000 | 1.3838 | 0.5000 |
| original | 19 | expert_ablation | 2 | 0.0000 | 0.0000 | 1.3839 | 0.5000 |
| original | 19 | expert_ablation | 3 | 0.0000 | 0.0000 | 1.3839 | 0.5000 |
| original | 19 | router_override | most_used_pair | 0.0000 | 0.0000 | 0.6931 | 0.5000 |
| original | 19 | route_randomization | uniform_topk_random | 0.0000 | 0.0000 | 0.6931 | 0.5000 |
| fresh | 23 | baseline | - | 1.0000 | 0.9641 | 1.3804 | 0.5000 |
| fresh | 23 | expert_ablation | 0 | 0.0000 | 0.0000 | 1.3805 | 0.5000 |
| fresh | 23 | expert_ablation | 1 | 0.0000 | 0.0000 | 1.3807 | 0.5000 |
| fresh | 23 | expert_ablation | 2 | 0.0000 | 0.0000 | 1.3806 | 0.5000 |
| fresh | 23 | expert_ablation | 3 | 0.0000 | 0.0000 | 1.3806 | 0.5000 |
| fresh | 23 | router_override | most_used_pair | 0.0000 | 0.0000 | 0.6931 | 0.5000 |
| fresh | 23 | route_randomization | uniform_topk_random | 0.0000 | 0.0000 | 0.6931 | 0.5000 |
| fresh | 29 | baseline | - | 1.0000 | 0.9613 | 1.3778 | 0.5000 |
| fresh | 29 | expert_ablation | 0 | 0.0000 | 0.0000 | 1.3786 | 0.5000 |
| fresh | 29 | expert_ablation | 1 | 0.4219 | 0.4033 | 1.3786 | 0.5000 |
| fresh | 29 | expert_ablation | 2 | 0.0000 | 0.0000 | 1.3786 | 0.5000 |
| fresh | 29 | expert_ablation | 3 | 0.0000 | 0.0000 | 1.3790 | 0.5000 |
| fresh | 29 | router_override | most_used_pair | 0.0000 | 0.0000 | 0.6931 | 0.5000 |
| fresh | 29 | route_randomization | uniform_topk_random | 0.9844 | 0.8447 | 0.6931 | 0.5000 |

## Interpretation

- `fresh` seed `23` uses top experts `[3, 1]` for the fixed-router probe. Baseline greedy success is `1.0000`; fixed-router drop is `1.0000`, random-routing drop is `1.0000`, and worst single-expert ablation drop is `1.0000` (expert `0`).
- `fresh` seed `29` uses top experts `[3, 1]` for the fixed-router probe. Baseline greedy success is `1.0000`; fixed-router drop is `1.0000`, random-routing drop is `0.0156`, and worst single-expert ablation drop is `1.0000` (expert `0`).
- `original` seed `7` uses top experts `[3, 2]` for the fixed-router probe. Baseline greedy success is `1.0000`; fixed-router drop is `1.0000`, random-routing drop is `1.0000`, and worst single-expert ablation drop is `1.0000` (expert `0`).
- `original` seed `19` uses top experts `[1, 2]` for the fixed-router probe. Baseline greedy success is `0.5781`; fixed-router drop is `0.5781`, random-routing drop is `0.5781`, and worst single-expert ablation drop is `0.5781` (expert `0`).
- Learned routing assignments are causally relevant under this bounded probe family: disrupting routing materially degrades greedy DoorKey success.
