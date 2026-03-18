# Causal Route-Dependence Report

- external evaluation episodes per probe: `64`
- git commit: `21a890285831feb89c6de03ea16619239ff732e3`
- git dirty: `True`

| Lane | Seed | Probe | Detail | Greedy Success | Greedy Return | Route Entropy | Active Compute |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| post_pass_f | 137 | baseline | - | 1.0000 | 0.9653 | 1.3823 | 0.5000 |
| post_pass_f | 137 | expert_ablation | 0 | 0.1250 | 0.1208 | 1.3827 | 0.5000 |
| post_pass_f | 137 | expert_ablation | 1 | 0.0000 | 0.0000 | 1.3829 | 0.5000 |
| post_pass_f | 137 | expert_ablation | 2 | 0.0000 | 0.0000 | 1.3826 | 0.5000 |
| post_pass_f | 137 | expert_ablation | 3 | 0.0000 | 0.0000 | 1.3828 | 0.5000 |
| post_pass_f | 137 | router_override | most_used_pair | 0.0000 | 0.0000 | 0.6931 | 0.5000 |
| post_pass_f | 137 | route_randomization | uniform_topk_random | 0.0156 | 0.0069 | 0.6931 | 0.5000 |
| fresh_final | 47 | baseline | - | 1.0000 | 0.9629 | 1.3617 | 0.5000 |
| fresh_final | 47 | expert_ablation | 0 | 0.0000 | 0.0000 | 1.3655 | 0.5000 |
| fresh_final | 47 | expert_ablation | 1 | 0.0000 | 0.0000 | 1.3655 | 0.5000 |
| fresh_final | 47 | expert_ablation | 2 | 1.0000 | 0.9559 | 1.3625 | 0.5000 |
| fresh_final | 47 | expert_ablation | 3 | 0.0000 | 0.0000 | 1.3650 | 0.5000 |
| fresh_final | 47 | router_override | most_used_pair | 0.0000 | 0.0000 | 0.6931 | 0.5000 |
| fresh_final | 47 | route_randomization | uniform_topk_random | 0.0000 | 0.0000 | 0.6931 | 0.5000 |
| post_pass_a | 67 | baseline | - | 1.0000 | 0.9648 | 1.3772 | 0.5000 |
| post_pass_a | 67 | expert_ablation | 0 | 0.0000 | 0.0000 | 1.3779 | 0.5000 |
| post_pass_a | 67 | expert_ablation | 1 | 0.0000 | 0.0000 | 1.3779 | 0.5000 |
| post_pass_a | 67 | expert_ablation | 2 | 0.0000 | 0.0000 | 1.3780 | 0.5000 |
| post_pass_a | 67 | expert_ablation | 3 | 0.2031 | 0.1970 | 1.3773 | 0.5000 |
| post_pass_a | 67 | router_override | most_used_pair | 0.0000 | 0.0000 | 0.6931 | 0.5000 |
| post_pass_a | 67 | route_randomization | uniform_topk_random | 0.0000 | 0.0000 | 0.6931 | 0.5000 |

## Interpretation

- `fresh_final` seed `47` uses top experts `[3, 1]` for the fixed-router probe. Baseline greedy success is `1.0000`; fixed-router drop is `1.0000`, random-routing drop is `1.0000`, and worst single-expert ablation drop is `1.0000` (expert `0`).
- `post_pass_a` seed `67` uses top experts `[1, 2]` for the fixed-router probe. Baseline greedy success is `1.0000`; fixed-router drop is `1.0000`, random-routing drop is `1.0000`, and worst single-expert ablation drop is `1.0000` (expert `0`).
- `post_pass_f` seed `137` uses top experts `[3, 2]` for the fixed-router probe. Baseline greedy success is `1.0000`; fixed-router drop is `1.0000`, random-routing drop is `0.9844`, and worst single-expert ablation drop is `1.0000` (expert `1`).
- Learned routing assignments are causally relevant under this bounded probe family: disrupting routing materially degrades greedy DoorKey success.
