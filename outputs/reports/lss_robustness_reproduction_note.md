# Learner-State Robustness Reproduction Note

- output root: `outputs/reproductions/lss_robustness_baseline`
- external evaluation episodes per mode: `64`
- git commit: `dec7eb0143dda84877001af20d8f97e2d92c540f`
- git dirty: `True`

## Per-Seed Results

| Seed | Variant | Greedy Success | Best Sampled Success | Best Sampled Mode | Config | Checkpoint | Command |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 7 | flat_dense | 1.0000 | 1.0000 | sampled_t1.0 | `outputs/reproductions/lss_robustness_baseline/seed_7/flat_dense_ent1e3/resolved_config.yaml` | `outputs/reproductions/lss_robustness_baseline/seed_7/flat_dense_ent1e3/latest.pt` | `torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch --config outputs/reproductions/lss_robustness_baseline/seed_7/configs/flat_dense_ent1e3.yaml --device auto` |
| 7 | recovered token_dense | 0.7031 | 1.0000 | sampled_t1.0 | `outputs/reproductions/lss_robustness_baseline/seed_7/token_dense_ent1e3/resolved_config.yaml` | `outputs/reproductions/lss_robustness_baseline/seed_7/token_dense_ent1e3/latest.pt` | `torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch --config outputs/reproductions/lss_robustness_baseline/seed_7/configs/token_dense_ent1e3.yaml --device auto` |
| 7 | baseline SARE | 0.0000 | 1.0000 | sampled_t1.0 | `outputs/reproductions/lss_robustness_baseline/seed_7/sare_ent1e3/resolved_config.yaml` | `outputs/reproductions/lss_robustness_baseline/seed_7/sare_ent1e3/latest.pt` | `torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch --config outputs/reproductions/lss_robustness_baseline/seed_7/configs/sare_ent1e3.yaml --device auto` |
| 7 | learner-state SARE | 0.5000 | 0.5625 | sampled_t1.0 | `outputs/reproductions/lss_robustness_baseline/seed_7/flat_dense_to_sare_lss/resolved_config.yaml` | `outputs/reproductions/lss_robustness_baseline/seed_7/flat_dense_to_sare_lss/latest.pt` | `python -m psmn_rl.analysis.learner_state_supervision run --spec outputs/reproductions/lss_robustness_baseline/seed_7/configs/flat_dense_to_sare_lss.yaml --device auto` |
| 11 | flat_dense | 1.0000 | 1.0000 | sampled_t1.0 | `outputs/reproductions/lss_robustness_baseline/seed_11/flat_dense_ent1e3/resolved_config.yaml` | `outputs/reproductions/lss_robustness_baseline/seed_11/flat_dense_ent1e3/latest.pt` | `torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch --config outputs/reproductions/lss_robustness_baseline/seed_11/configs/flat_dense_ent1e3.yaml --device auto` |
| 11 | recovered token_dense | 0.0000 | 0.0000 | sampled_t1.0 | `outputs/reproductions/lss_robustness_baseline/seed_11/token_dense_ent1e3/resolved_config.yaml` | `outputs/reproductions/lss_robustness_baseline/seed_11/token_dense_ent1e3/latest.pt` | `torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch --config outputs/reproductions/lss_robustness_baseline/seed_11/configs/token_dense_ent1e3.yaml --device auto` |
| 11 | baseline SARE | 0.0000 | 0.7812 | sampled_t1.0 | `outputs/reproductions/lss_robustness_baseline/seed_11/sare_ent1e3/resolved_config.yaml` | `outputs/reproductions/lss_robustness_baseline/seed_11/sare_ent1e3/latest.pt` | `torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch --config outputs/reproductions/lss_robustness_baseline/seed_11/configs/sare_ent1e3.yaml --device auto` |
| 11 | learner-state SARE | 1.0000 | 1.0000 | sampled_t1.0 | `outputs/reproductions/lss_robustness_baseline/seed_11/flat_dense_to_sare_lss/resolved_config.yaml` | `outputs/reproductions/lss_robustness_baseline/seed_11/flat_dense_to_sare_lss/latest.pt` | `python -m psmn_rl.analysis.learner_state_supervision run --spec outputs/reproductions/lss_robustness_baseline/seed_11/configs/flat_dense_to_sare_lss.yaml --device auto` |
| 19 | flat_dense | 1.0000 | 1.0000 | sampled_t1.0 | `outputs/reproductions/lss_robustness_baseline/seed_19/flat_dense_ent1e3/resolved_config.yaml` | `outputs/reproductions/lss_robustness_baseline/seed_19/flat_dense_ent1e3/latest.pt` | `torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch --config outputs/reproductions/lss_robustness_baseline/seed_19/configs/flat_dense_ent1e3.yaml --device auto` |
| 19 | recovered token_dense | 1.0000 | 1.0000 | sampled_t1.0 | `outputs/reproductions/lss_robustness_baseline/seed_19/token_dense_ent1e3/resolved_config.yaml` | `outputs/reproductions/lss_robustness_baseline/seed_19/token_dense_ent1e3/latest.pt` | `torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch --config outputs/reproductions/lss_robustness_baseline/seed_19/configs/token_dense_ent1e3.yaml --device auto` |
| 19 | baseline SARE | 0.0000 | 0.9688 | sampled_t1.0 | `outputs/reproductions/lss_robustness_baseline/seed_19/sare_ent1e3/resolved_config.yaml` | `outputs/reproductions/lss_robustness_baseline/seed_19/sare_ent1e3/latest.pt` | `torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch --config outputs/reproductions/lss_robustness_baseline/seed_19/configs/sare_ent1e3.yaml --device auto` |
| 19 | learner-state SARE | 0.0000 | 0.0000 | sampled_t1.0 | `outputs/reproductions/lss_robustness_baseline/seed_19/flat_dense_to_sare_lss/resolved_config.yaml` | `outputs/reproductions/lss_robustness_baseline/seed_19/flat_dense_to_sare_lss/latest.pt` | `python -m psmn_rl.analysis.learner_state_supervision run --spec outputs/reproductions/lss_robustness_baseline/seed_19/configs/flat_dense_to_sare_lss.yaml --device auto` |

## Interpretation

- This note reproduces the current multi-seed DoorKey teacher/student baseline on the external 64-episode decision path.
- The routed comparison is only meaningful if these numbers stay close to the published teacher-extraction lane.
