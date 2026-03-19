# Portfolio Campaign Registration

- active incumbent: `round6`
- active benchmark pack before this program: `outputs/reports/expansion_mega_program_candidate_pack.json`
- archived frozen pack: `outputs/reports/frozen_benchmark_pack.json`
- git commit: `85e2aaadae7c741e625431dbf494372007eb001c`
- git dirty: `True`

## 50/50 Portfolio Split

- fruitful-track challenger count: `20`
- exploratory-track challenger count: `20`
- total challenger count: `40`

## Families

- development families: `[{'lane': 'prospective_c', 'seeds': [181, 191, 193]}, {'lane': 'prospective_d', 'seeds': [197, 199, 211]}, {'lane': 'prospective_f', 'seeds': [233, 239, 241]}]`
- holdout families: `[{'lane': 'prospective_g', 'seeds': [251, 257, 263]}, {'lane': 'prospective_h', 'seeds': [269, 271, 277]}, {'lane': 'prospective_i', 'seeds': [281, 283, 293]}, {'lane': 'prospective_j', 'seeds': [307, 311, 313]}]`
- healthy anti-regression families: `[{'lane': 'original', 'seeds': [7, 11, 19]}, {'lane': 'fresh', 'seeds': [23, 29, 31]}, {'lane': 'fresh_extra', 'seeds': [37, 41, 43]}]`
- exploratory adjacent-task track: `[{'lane': 'keycorridor', 'seeds': [7, 11, 19]}]`
- family counts: `{'near_neighbor_rounds': 6, 'near_neighbor_temperature': 3, 'hard_postunlock_weighting': 2, 'hard_mixed_weighting': 6, 'hard_smoothing_postunlock': 3, 'teacher_smoothing': 3, 'confidence_weighting': 6, 'bridge_weighting': 6, 'replay_cap_bridge_revisit': 5}`

## Program Size

- distinct mechanism directions: `15`
- mechanism direction names: `['cap_balanced_recent_replay', 'cap_recent_replay', 'late_round_carry_key_bridge', 'late_round_locked_door_bridge', 'late_round_mixed_post_unlock_disagreement', 'late_round_phase_balanced_replay', 'late_round_post_unlock_weight', 'late_round_selection_late', 'late_round_selection_mid', 'late_round_selection_plus1', 'late_round_teacher_confidence_weighting', 'late_round_teacher_temperature_090', 'mild_teacher_smoothing', 'mild_teacher_smoothing_plus_post_unlock', 'round_selection_round5']`
- Stage 1 fruitful top-k: `3`
- Stage 1 exploratory top-k: `3`
- verification reruns per survivor: `2`

## Fair-Shot Rule

- Each family gets a bounded calibration sweep before being declared alive or dead.
- A family may be pruned early only if it is catastrophically below `round6` on multiple development families and the rerun confirms the failure.
- No family is promoted on one lucky family group or one lucky seed.

## Historical Context

- `outputs/reports/long_campaign_decision_memo.md`
- `outputs/reports/post_pass_canonization_decision_memo.md`
- `outputs/reports/canonization_decision_memo.md`
- `outputs/reports/hard_family_canonization_decision_memo.md`
- `outputs/reports/hard_family_saturation_decision_memo.md`
- `outputs/reports/successor_validation_decision_memo.md`
- `outputs/reports/successor_challenge_decision_memo.md`
- `outputs/reports/successor_stress_decision_memo.md`
- `outputs/reports/successor_stress_followup_decision_memo.md`
- `outputs/reports/successor_stress_extended_decision_memo.md`
- `outputs/reports/successor_tiebreak_decision_memo.md`
- `outputs/reports/successor_migration_decision_memo.md`
- `outputs/reports/successor_mega_league_decision_memo.md`
- `outputs/reports/expansion_mega_program_decision_memo.md`
