# Portfolio Active-State Doctor

- git commit: `9bb35f0bda056ed5e24b3a3b0aa16c706700633e`
- git dirty: `True`
- overall: `pass`

## Checks

| Check | Status | Detail |
| --- | --- | --- |
| `candidate_round6` | `pass` | candidate_name=round6 |
| `candidate_pack_current` | `pass` | active_canonical_pack=outputs/reports/portfolio_candidate_pack.json |
| `candidate_archived_pack_frozen` | `pass` | archived_legacy_pack=outputs/reports/frozen_benchmark_pack.json, frozen_pack_reference=outputs/reports/frozen_benchmark_pack.json |
| `candidate_eval_doorkey_external64` | `pass` | task=DoorKey, evaluation.task=DoorKey, path_key=external_policy_diagnostics, episodes=64 |
| `gate_targets_current_active_pack` | `pass` | candidate_pack=outputs/reports/portfolio_candidate_pack.json, frozen_pack=outputs/reports/frozen_benchmark_pack.json |
| `gate_pack_mode_pass` | `pass` | mode=pack, verdict=PASS: thaw consideration allowed |
| `gate_combined_picture_pass` | `pass` | combined_picture_mean=PASS, combined_picture_failures=PASS |
| `contract_matches_active_roles` | `pass` | active_candidate=round6, default_restart_prior=round7, replay_validated_alternate=round10 |
| `frontier_doctor_pass` | `pass` | frontier_doctor.overall=pass |
