# Portfolio Campaign Registration

- active incumbent: `round6`
- active benchmark pack before this program: `outputs/reports/portfolio_candidate_pack.json`
- archived frozen pack: `outputs/reports/frozen_benchmark_pack.json`
- git commit: `251c92bc33d7b9ec5aa313dcbdd1a9c245bfa7fe`
- git dirty: `True`

## 50/50 Portfolio Split

- fruitful-track challenger count: `24`
- exploratory-track challenger count: `24`
- total challenger count: `48`

## Families

- development families: `[{'lane': 'prospective_c', 'seeds': [181, 191, 193]}, {'lane': 'prospective_d', 'seeds': [197, 199, 211]}, {'lane': 'prospective_e', 'seeds': [223]}, {'lane': 'prospective_f', 'seeds': [233, 239, 241]}]`
- holdout families: `[{'lane': 'prospective_g', 'seeds': [251, 257, 263]}, {'lane': 'prospective_h', 'seeds': [269, 271, 277]}, {'lane': 'prospective_i', 'seeds': [281, 283, 293]}, {'lane': 'prospective_j', 'seeds': [307, 311, 313]}]`
- healthy anti-regression families: `[{'lane': 'original', 'seeds': [7, 11, 19]}, {'lane': 'fresh', 'seeds': [23, 29, 31]}, {'lane': 'fresh_extra', 'seeds': [37, 41, 43]}, {'lane': 'fresh_final', 'seeds': [47, 53, 59]}]`
- hard-seed / hard-pattern families: `[{'lane': 'prospective_c', 'seeds': [193]}, {'lane': 'prospective_f', 'seeds': [233]}, {'lane': 'prospective_h', 'seeds': [269]}, {'lane': 'prospective_h', 'seeds': [277]}]`
- exploratory adjacent-task track: `[{'lane': 'keycorridor', 'seeds': [7, 11, 19]}]`
- family counts: `{'near_neighbor_rounds': 6, 'frontier_prior_confirmation': 1, 'near_neighbor_temperature': 2, 'hard_postunlock_weighting': 3, 'hard_mixed_weighting': 4, 'hard_smoothing_postunlock': 1, 'frontier_restart_policy': 1, 'selection_tiebreak': 6, 'rsm_detached_warmup_final_step': 5, 'rsm_stochastic_credit_horizon': 5, 'settling_diagnostics': 4, 'confidence_weighting': 5, 'replay_cap_bridge_revisit': 5}`

## Program Size

- distinct mechanism directions: `29`
- mechanism direction names: `['base_round4_restart_prior', 'cap_balanced_recent_replay', 'confidence_weighting_plus_smoothing', 'late_round_carry_key_bridge_stronger', 'late_round_locked_door_bridge_stronger', 'late_round_mixed_post_unlock_disagreement', 'late_round_phase_balanced_replay', 'late_round_post_unlock_weight', 'late_round_selection_late', 'late_round_selection_mid', 'late_round_selection_plus1', 'late_round_selection_plus5', 'late_round_teacher_confidence_weighting', 'mild_teacher_smoothing_frontier_prior', 'mild_teacher_smoothing_plus_post_unlock', 'post_unlock_rebalance_round7', 'round_selection_round5', 'selection_final_greedy', 'selection_mixed_weighting', 'selection_post_unlock_weight', 'selection_smoothing_frontier', 'settling_checkpoint_proxy_last_step', 'settling_checkpoint_proxy_last_two', 'settling_checkpoint_proxy_stochastic', 'stochastic_last_step_last_two_mix', 'stochastic_last_step_last_two_mix_025', 'stochastic_last_step_last_two_mix_075', 'warmup_one_final_step_only', 'warmup_two_final_step_only']`
- Stage 1 fruitful top-k: `4`
- Stage 1 exploratory top-k: `4`
- verification reruns per survivor: `2`
- route-validation cases: `{'dev': {'lane': 'prospective_d', 'seed': 197, 'stage_key': 'stage1_screening'}, 'holdout': {'lane': 'prospective_h', 'seed': 269, 'stage_key': 'stage3_holdout'}, 'healthy': {'lane': 'fresh', 'seed': 23, 'stage_key': 'stage4_antiregression'}, 'hard': {'lane': 'prospective_f', 'seed': 233, 'stage_key': 'stage1_screening'}}`
- stability cases: `{'dev': {'lane': 'prospective_d', 'seed': 199, 'stage_key': 'stage1_screening'}, 'holdout': {'lane': 'prospective_g', 'seed': 251, 'stage_key': 'stage3_holdout'}, 'healthy': {'lane': 'fresh_extra', 'seed': 37, 'stage_key': 'stage4_antiregression'}, 'hard': {'lane': 'prospective_h', 'seed': 277, 'stage_key': 'stage3_holdout'}}`

## Fair-Shot Rule

- Each family gets a bounded calibration sweep before being declared alive or dead.
- A family may be pruned early only if it is catastrophically below `round6` on multiple development families and the rerun confirms the failure.
- No family is promoted on one lucky family group or one lucky seed.

## Exact Candidate Families

### exploit

- `frontier_prior_challengers`: `round7`, `round8`, `round9`, `round10`, `round11`, `round12`, `round7_temp_095`, `round10_temp_095`
  rationale: closest measured near-neighbors around round count and mild smoothing, targeted at the repeated 0.9000 parity bucket
- `hard_block_parity_breakers`: `round7_post_unlock_x5`, `round10_post_unlock_x5`, `round12_post_unlock_x5`, `round10_post_unlock_x4_dis025`, `round12_post_unlock_x4_dis025`, `round10_post_unlock_x5_dis025`, `round12_post_unlock_x5_dis025`, `round10_temp095_post5`
  rationale: post-unlock and mixed-weight variants aimed directly at the late-collapse blocker family
- `restart_policy_exploitation`: `round5`, `post_unlock_weighted`
  rationale: explicit re-check of the live restart schedule against the hold-only and legacy base-restart priors
- `candidate_selection_exploit_lane`: `select_round7_final`, `select_round10_final`, `select_round7_temp095`, `select_round10_temp095`, `select_round7_post5`, `select_round10_mix025`
  rationale: selection rules that could turn repeated raw ties into a reproducible post-control edge without changing the underlying family

### explore

- `detached_warmup_terminal_supervision`: `round7_warm1_last_step`, `round10_warm1_last_step`, `round12_warm1_last_step`, `round10_warm2_last_step`, `round12_warm2_last_step`
  rationale: bounded warm-up and terminal-credit variants testing whether a more depth-agnostic operator helps
- `stochastic_credit_horizon`: `round7_stoch50_last_two`, `round10_stoch50_last_two`, `round12_stoch50_last_two`, `round10_stoch25_last_two`, `round10_stoch75_last_two`
  rationale: explicit last-step versus last-two-step mixtures to test recursive credit assignment bottlenecks
- `settling_diagnostics`: `settling_round7_last_step`, `settling_round10_last_step`, `settling_round10_last_two`, `settling_round10_stoch50`
  rationale: settling-aware selection proxies on the most plausible warm-up and stochastic lineages
- `confidence_disagreement_shaping`: `round7_conf_post4`, `round10_conf_post4`, `round12_conf_post4`, `round10_conf_temp095_post4`, `round12_conf_temp095_post4`
  rationale: confidence-weighted KL with and without mild smoothing, targeted at the remaining parity cluster
- `replay_cap_bridge_revisit`: `round7_phase_balanced_4096`, `round10_phase_balanced_4096`, `round10_cap_balanced_4096`, `round10_carry3_post5`, `round10_door3_post5`
  rationale: fair-shot revisit of previously weak replay and bridge ideas under tighter candidate selection

## Seed-Block Rationale

- `dev`: prospective_c/prospective_d/prospective_e/prospective_f cover the known parity bucket, the late-collapse blocker, and one broader stress lane used for selection.
- `holdout`: prospective_g/prospective_h/prospective_i/prospective_j stay out of Stage 2 selection and test whether any candidate generalizes beyond the dev blocker surface.
- `healthy`: original/fresh/fresh_extra/fresh_final preserve the known healthy DoorKey behavior that `round6` must not break to look better.
- `hard_block`: prospective_c/193, prospective_f/233, prospective_h/269, and prospective_h/277 are the measured blocker quartet carried forward from the weakness dossier and triage stack.
- `route_and_stability`: dev, holdout, healthy, and hard-block seeds are all retained in route/stability so any surviving story has to stay routed and non-brittle across the full narrowed state.

## Explicit Omissions

- The current learner-state supervision and evaluation surface does not expose an independent test-time recursion-depth override, so the train-shallow/test-deep idea is approximated here with settling-aware and late-round proxy lanes rather than a fake unsupported knob.

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
- `outputs/reports/portfolio_decision_memo.md`
- `outputs/reports/portfolio_frontier_manifest.md`
- `outputs/reports/portfolio_operational_state.md`
