# Causal Route-Dependence Report

- external evaluation episodes per probe: `64`
- git commit: `21a890285831feb89c6de03ea16619239ff732e3`
- git dirty: `True`

| Lane | Seed | Probe | Detail | Greedy Success | Greedy Return | Route Entropy | Active Compute |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| prospective_b | 173 | baseline | - | 1.0000 | 0.9644 | 1.3795 | 0.5000 |
| prospective_b | 173 | expert_ablation | 0 | 0.0000 | 0.0000 | 1.3803 | 0.5000 |
| prospective_b | 173 | expert_ablation | 1 | 0.5156 | 0.4959 | 1.3793 | 0.5000 |
| prospective_b | 173 | expert_ablation | 2 | 0.0000 | 0.0000 | 1.3800 | 0.5000 |
| prospective_b | 173 | expert_ablation | 3 | 0.2188 | 0.2109 | 1.3807 | 0.5000 |
| prospective_b | 173 | router_override | most_used_pair | 0.0000 | 0.0000 | 0.6931 | 0.5000 |
| prospective_b | 173 | route_randomization | uniform_topk_random | 0.0469 | 0.0155 | 0.6931 | 0.5000 |

## Interpretation

- `prospective_b` seed `173` uses top experts `[2, 0]` for the fixed-router probe. Baseline greedy success is `1.0000`; fixed-router drop is `1.0000`, random-routing drop is `0.9531`, and worst single-expert ablation drop is `1.0000` (expert `0`).
- Learned routing assignments are causally relevant under this bounded probe family: disrupting routing materially degrades greedy DoorKey success.
