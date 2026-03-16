# Distilled Route Integrity Report

- episodes: `64`

| Run | Variant | Eval Success | Eval Return | Route Entropy | Path Entropy | Active Compute | expert_load_0 | expert_load_1 | expert_load_2 | expert_load_3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| minigrid_doorkey_sare_ent1e3 | sare | 0.0000 | 0.0000 | 1.3855 | 1.2264 | 0.5000 | 0.2552 | 0.2484 | 0.2480 | 0.2484 |
| flat_dense_to_sare_lss | sare | 0.5312 | 0.5124 | 1.3821 | 1.2931 | 0.5000 | 0.2499 | 0.2578 | 0.2488 | 0.2435 |

## Interpretation

- `flat_dense_to_sare_lss` changes greedy success from `0.0000` to `0.5312`.
- Route entropy moves from `1.3855` to `1.3821`, and active compute proxy moves from `0.5000` to `0.5000`.
