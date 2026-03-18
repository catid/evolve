# Successor Migration Registration

- archived frozen benchmark pack: `outputs/reports/frozen_benchmark_pack.json`
- current qualified successor pack: `outputs/reports/hard_family_saturation_successor_pack.json`
- current qualified successor: `round6`
- git commit: `6e7277a8d8489cb74927ff7c7b3e072809491bab`
- git dirty: `True`

## Splits

- challenger development families: `[{'lane': 'prospective_g', 'seeds': [251, 257, 263]}, {'lane': 'prospective_h', 'seeds': [269, 271, 277]}]`
- challenger holdout families: `[{'lane': 'prospective_i', 'seeds': [281, 283, 293]}, {'lane': 'prospective_j', 'seeds': [307, 311, 313]}]`
- healthy anti-regression families: `[{'lane': 'original', 'seeds': [7, 11, 19]}, {'lane': 'fresh', 'seeds': [23, 29, 31]}, {'lane': 'fresh_extra', 'seeds': [37, 41, 43]}]`

## Challenger League

- mechanism directions: `['carry_key_bridge_weighting', 'cleanup_round_extension', 'cleanup_round_phase_balanced_disagreement', 'disagreement_threshold_weighting', 'locked_door_bridge_weighting', 'post_unlock_weight_schedule']`
- total challenger variants: `12`

## Migration Rule

- If no challenger survives the full league cleanly, the program still continues through route, stability, and pack/gate checks for round6 before deciding active canonization vs no migration.
- If a challenger survives the full league and still beats or equals round6 after controls, holdout, anti-regression, route, and stability, the challenger is packaged for migration instead.

## Historical Context

- `outputs/reports/successor_validation_decision_memo.md`
- `outputs/reports/successor_challenge_decision_memo.md`
- `outputs/reports/successor_stress_decision_memo.md`
- `outputs/reports/successor_stress_followup_decision_memo.md`
- `outputs/reports/successor_stress_extended_decision_memo.md`
- `outputs/reports/successor_tiebreak_decision_memo.md`
