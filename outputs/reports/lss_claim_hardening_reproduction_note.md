# Claim-Hardening Reproduction Note

- baseline root: `outputs/reproductions/lss_robustness_baseline`
- improved root: `outputs/reproductions/lss_claim_hardening_baseline`
- external evaluation episodes per mode: `64`
- git commit: `42fc49b00a185b46b723a326ace904654678e505`
- git dirty: `True`

## Per-Seed Results

| Seed | Variant | Greedy Success | Best Sampled Success | Best Sampled Mode | Config | Checkpoint | Command |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 7 | flat_dense | 1.0000 | 1.0000 | sampled_t1.0 | `outputs/reproductions/lss_robustness_baseline/seed_7/flat_dense_ent1e3/resolved_config.yaml` | `outputs/reproductions/lss_robustness_baseline/seed_7/flat_dense_ent1e3/latest.pt` | `torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch --config outputs/reproductions/lss_robustness_baseline/seed_7/configs/flat_dense_ent1e3.yaml --device auto` |
| 7 | recovered token_dense | 0.7031 | 1.0000 | sampled_t1.0 | `outputs/reproductions/lss_robustness_baseline/seed_7/token_dense_ent1e3/resolved_config.yaml` | `outputs/reproductions/lss_robustness_baseline/seed_7/token_dense_ent1e3/latest.pt` | `torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch --config outputs/reproductions/lss_robustness_baseline/seed_7/configs/token_dense_ent1e3.yaml --device auto` |
| 7 | baseline PPO SARE | 0.0000 | 1.0000 | sampled_t1.0 | `outputs/reproductions/lss_robustness_baseline/seed_7/sare_ent1e3/resolved_config.yaml` | `outputs/reproductions/lss_robustness_baseline/seed_7/sare_ent1e3/latest.pt` | `torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch --config outputs/reproductions/lss_robustness_baseline/seed_7/configs/sare_ent1e3.yaml --device auto` |
| 7 | KL learner-state SARE | 1.0000 | 1.0000 | sampled_t1.0 | `outputs/reproductions/lss_claim_hardening_baseline/seed_7/kl_lss_sare/student_resolved_config.yaml` | `outputs/reproductions/lss_claim_hardening_baseline/seed_7/kl_lss_sare/latest.pt` | `python -m psmn_rl.analysis.learner_state_supervision run --spec outputs/reproductions/lss_claim_hardening_baseline/seed_7/configs/kl_lss_sare.yaml --device auto` |
| 11 | flat_dense | 1.0000 | 1.0000 | sampled_t1.0 | `outputs/reproductions/lss_robustness_baseline/seed_11/flat_dense_ent1e3/resolved_config.yaml` | `outputs/reproductions/lss_robustness_baseline/seed_11/flat_dense_ent1e3/latest.pt` | `torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch --config outputs/reproductions/lss_robustness_baseline/seed_11/configs/flat_dense_ent1e3.yaml --device auto` |
| 11 | recovered token_dense | 0.0000 | 0.0000 | sampled_t1.0 | `outputs/reproductions/lss_robustness_baseline/seed_11/token_dense_ent1e3/resolved_config.yaml` | `outputs/reproductions/lss_robustness_baseline/seed_11/token_dense_ent1e3/latest.pt` | `torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch --config outputs/reproductions/lss_robustness_baseline/seed_11/configs/token_dense_ent1e3.yaml --device auto` |
| 11 | baseline PPO SARE | 0.0000 | 0.7812 | sampled_t1.0 | `outputs/reproductions/lss_robustness_baseline/seed_11/sare_ent1e3/resolved_config.yaml` | `outputs/reproductions/lss_robustness_baseline/seed_11/sare_ent1e3/latest.pt` | `torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch --config outputs/reproductions/lss_robustness_baseline/seed_11/configs/sare_ent1e3.yaml --device auto` |
| 11 | KL learner-state SARE | 0.5625 | 0.9219 | sampled_t1.0 | `outputs/reproductions/lss_claim_hardening_baseline/seed_11/kl_lss_sare/student_resolved_config.yaml` | `outputs/reproductions/lss_claim_hardening_baseline/seed_11/kl_lss_sare/latest.pt` | `python -m psmn_rl.analysis.learner_state_supervision run --spec outputs/reproductions/lss_claim_hardening_baseline/seed_11/configs/kl_lss_sare.yaml --device auto` |
| 19 | flat_dense | 1.0000 | 1.0000 | sampled_t1.0 | `outputs/reproductions/lss_robustness_baseline/seed_19/flat_dense_ent1e3/resolved_config.yaml` | `outputs/reproductions/lss_robustness_baseline/seed_19/flat_dense_ent1e3/latest.pt` | `torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch --config outputs/reproductions/lss_robustness_baseline/seed_19/configs/flat_dense_ent1e3.yaml --device auto` |
| 19 | recovered token_dense | 1.0000 | 1.0000 | sampled_t1.0 | `outputs/reproductions/lss_robustness_baseline/seed_19/token_dense_ent1e3/resolved_config.yaml` | `outputs/reproductions/lss_robustness_baseline/seed_19/token_dense_ent1e3/latest.pt` | `torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch --config outputs/reproductions/lss_robustness_baseline/seed_19/configs/token_dense_ent1e3.yaml --device auto` |
| 19 | baseline PPO SARE | 0.0000 | 0.9688 | sampled_t1.0 | `outputs/reproductions/lss_robustness_baseline/seed_19/sare_ent1e3/resolved_config.yaml` | `outputs/reproductions/lss_robustness_baseline/seed_19/sare_ent1e3/latest.pt` | `torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch --config outputs/reproductions/lss_robustness_baseline/seed_19/configs/sare_ent1e3.yaml --device auto` |
| 19 | KL learner-state SARE | 0.5781 | 0.6406 | sampled_t1.0 | `outputs/reproductions/lss_claim_hardening_baseline/seed_19/kl_lss_sare/student_resolved_config.yaml` | `outputs/reproductions/lss_claim_hardening_baseline/seed_19/kl_lss_sare/latest.pt` | `python -m psmn_rl.analysis.learner_state_supervision run --spec outputs/reproductions/lss_claim_hardening_baseline/seed_19/configs/kl_lss_sare.yaml --device auto` |

## Interpretation

- This note re-evaluates the current DoorKey teacher/student baseline on the external 64-episode decision path before any new seed or matched-control extension.
- Forward work in this phase is only valid if the reproduced KL learner-state SARE lane stays close to the published robustness artifacts.
