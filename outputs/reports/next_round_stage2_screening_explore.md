# Portfolio Stage 1 Exploratory Screening

- track budget: `24` candidates
- advancing challengers from this track: `[]`
- incumbent round6 dev SARE/token/single: `0.9167` / `0.8854` / `0.9167`
- family counts: `{'replay_cap_bridge_revisit': 5, 'confidence_weighting': 5, 'rsm_stochastic_credit_horizon': 5, 'rsm_detached_warmup_final_step': 5, 'settling_diagnostics': 4}`

| Candidate | Family | Direction | Dev Mean | Δ vs round6 | Candidate-token | Candidate-single | Failures | Observed Specs | Stage 1 | Reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `round10_carry3_post5` | `replay_cap_bridge_revisit` | `late_round_carry_key_bridge_stronger` | `0.9000` | `-0.0167` | `0.0146` | `-0.0167` | `1` | `10` / `10` | `stop` | `stop: below incumbent dev mean` |
| `round10_conf_post4` | `confidence_weighting` | `late_round_teacher_confidence_weighting` | `0.9000` | `-0.0167` | `0.0146` | `-0.0167` | `1` | `10` / `10` | `stop` | `stop: below incumbent dev mean` |
| `round10_conf_temp095_post4` | `confidence_weighting` | `confidence_weighting_plus_smoothing` | `0.9000` | `-0.0167` | `0.0146` | `-0.0167` | `1` | `10` / `10` | `stop` | `stop: below incumbent dev mean` |
| `round10_door3_post5` | `replay_cap_bridge_revisit` | `late_round_locked_door_bridge_stronger` | `0.9000` | `-0.0167` | `0.0146` | `-0.0167` | `1` | `10` / `10` | `stop` | `stop: below incumbent dev mean` |
| `round12_conf_post4` | `confidence_weighting` | `late_round_teacher_confidence_weighting` | `0.9000` | `-0.0167` | `0.0146` | `-0.0167` | `1` | `10` / `10` | `stop` | `stop: below incumbent dev mean` |
| `round12_conf_temp095_post4` | `confidence_weighting` | `confidence_weighting_plus_smoothing` | `0.9000` | `-0.0167` | `0.0146` | `-0.0167` | `1` | `10` / `10` | `stop` | `stop: below incumbent dev mean` |
| `round12_warm1_last_step` | `rsm_detached_warmup_final_step` | `warmup_one_final_step_only` | `0.9000` | `-0.0167` | `0.0146` | `-0.0167` | `1` | `10` / `10` | `stop` | `stop: below incumbent dev mean` |
| `round7_conf_post4` | `confidence_weighting` | `late_round_teacher_confidence_weighting` | `0.9000` | `-0.0167` | `0.0146` | `-0.0167` | `1` | `10` / `10` | `stop` | `stop: below incumbent dev mean` |
| `settling_round10_last_two` | `settling_diagnostics` | `settling_checkpoint_proxy_last_two` | `0.9000` | `-0.0167` | `0.0146` | `-0.0167` | `1` | `10` / `10` | `stop` | `stop: below incumbent dev mean` |
| `round10_warm1_last_step` | `rsm_detached_warmup_final_step` | `warmup_one_final_step_only` | `0.8438` | `-0.0729` | `-0.0417` | `-0.0729` | `1` | `10` / `10` | `stop` | `stop: below incumbent dev mean` |
| `settling_round10_last_step` | `settling_diagnostics` | `settling_checkpoint_proxy_last_step` | `0.8438` | `-0.0729` | `-0.0417` | `-0.0729` | `1` | `10` / `10` | `stop` | `stop: below incumbent dev mean` |
| `round12_warm2_last_step` | `rsm_detached_warmup_final_step` | `warmup_two_final_step_only` | `0.8063` | `-0.1104` | `-0.0792` | `-0.1104` | `1` | `10` / `10` | `stop` | `stop: below incumbent dev mean` |
| `round10_stoch25_last_two` | `rsm_stochastic_credit_horizon` | `stochastic_last_step_last_two_mix_025` | `0.7453` | `-0.1714` | `-0.1401` | `-0.1714` | `2` | `10` / `10` | `stop` | `stop: new complete-seed failures` |
| `round10_stoch50_last_two` | `rsm_stochastic_credit_horizon` | `stochastic_last_step_last_two_mix` | `0.7141` | `-0.2026` | `-0.1714` | `-0.2026` | `1` | `10` / `10` | `stop` | `stop: below incumbent dev mean` |
| `round12_stoch50_last_two` | `rsm_stochastic_credit_horizon` | `stochastic_last_step_last_two_mix` | `0.7141` | `-0.2026` | `-0.1714` | `-0.2026` | `1` | `10` / `10` | `stop` | `stop: below incumbent dev mean` |
| `settling_round10_stoch50` | `settling_diagnostics` | `settling_checkpoint_proxy_stochastic` | `0.7141` | `-0.2026` | `-0.1714` | `-0.2026` | `1` | `10` / `10` | `stop` | `stop: below incumbent dev mean` |
| `round10_stoch75_last_two` | `rsm_stochastic_credit_horizon` | `stochastic_last_step_last_two_mix_075` | `0.6734` | `-0.2432` | `-0.2120` | `-0.2432` | `2` | `10` / `10` | `stop` | `stop: new complete-seed failures` |
| `round10_warm2_last_step` | `rsm_detached_warmup_final_step` | `warmup_two_final_step_only` | `0.5406` | `-0.3760` | `-0.3448` | `-0.3760` | `4` | `10` / `10` | `stop` | `stop: new complete-seed failures` |
| `round7_stoch50_last_two` | `rsm_stochastic_credit_horizon` | `stochastic_last_step_last_two_mix` | `0.4000` | `-0.5167` | `-0.4854` | `-0.5167` | `4` | `10` / `10` | `stop` | `stop: new complete-seed failures` |
| `round10_cap_balanced_4096` | `replay_cap_bridge_revisit` | `cap_balanced_recent_replay` | `0.3031` | `-0.6135` | `-0.5823` | `-0.6135` | `6` | `10` / `10` | `stop` | `stop: new complete-seed failures` |
| `round7_warm1_last_step` | `rsm_detached_warmup_final_step` | `warmup_one_final_step_only` | `0.1453` | `-0.7714` | `-0.7401` | `-0.7714` | `8` | `10` / `10` | `stop` | `stop: new complete-seed failures` |
| `settling_round7_last_step` | `settling_diagnostics` | `settling_checkpoint_proxy_last_step` | `0.1453` | `-0.7714` | `-0.7401` | `-0.7714` | `8` | `10` / `10` | `stop` | `stop: new complete-seed failures` |
| `round10_phase_balanced_4096` | `replay_cap_bridge_revisit` | `late_round_phase_balanced_replay` | `0.0938` | `-0.8229` | `-0.7917` | `-0.8229` | `7` | `10` / `10` | `stop` | `stop: new complete-seed failures` |
| `round7_phase_balanced_4096` | `replay_cap_bridge_revisit` | `late_round_phase_balanced_replay` | `0.0109` | `-0.9057` | `-0.8745` | `-0.9057` | `9` | `10` / `10` | `stop` | `stop: new complete-seed failures` |

| Candidate | Family | Block | Seed | Greedy Success | Δ vs round6 | Δ vs token_dense | Δ vs single_expert |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `round10_cap_balanced_4096` | `replay_cap_bridge_revisit` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_cap_balanced_4096` | `replay_cap_bridge_revisit` | prospective_c | 191 | 0.2812 | -0.7188 | -0.7188 | -0.7188 |
| `round10_cap_balanced_4096` | `replay_cap_bridge_revisit` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_cap_balanced_4096` | `replay_cap_bridge_revisit` | prospective_d | 197 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_cap_balanced_4096` | `replay_cap_bridge_revisit` | prospective_d | 199 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_cap_balanced_4096` | `replay_cap_bridge_revisit` | prospective_d | 211 | 0.7500 | -0.2500 | -0.2500 | -0.2500 |
| `round10_cap_balanced_4096` | `replay_cap_bridge_revisit` | prospective_e | 223 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_cap_balanced_4096` | `replay_cap_bridge_revisit` | prospective_f | 233 | 0.0000 | -1.0000 | -0.6250 | -1.0000 |
| `round10_cap_balanced_4096` | `replay_cap_bridge_revisit` | prospective_f | 239 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_cap_balanced_4096` | `replay_cap_bridge_revisit` | prospective_f | 241 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_carry3_post5` | `replay_cap_bridge_revisit` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_carry3_post5` | `replay_cap_bridge_revisit` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_carry3_post5` | `replay_cap_bridge_revisit` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_carry3_post5` | `replay_cap_bridge_revisit` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_carry3_post5` | `replay_cap_bridge_revisit` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_carry3_post5` | `replay_cap_bridge_revisit` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_carry3_post5` | `replay_cap_bridge_revisit` | prospective_e | 223 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_carry3_post5` | `replay_cap_bridge_revisit` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round10_carry3_post5` | `replay_cap_bridge_revisit` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_carry3_post5` | `replay_cap_bridge_revisit` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post4` | `confidence_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post4` | `confidence_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post4` | `confidence_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post4` | `confidence_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post4` | `confidence_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post4` | `confidence_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post4` | `confidence_weighting` | prospective_e | 223 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post4` | `confidence_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round10_conf_post4` | `confidence_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post4` | `confidence_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_temp095_post4` | `confidence_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_temp095_post4` | `confidence_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_temp095_post4` | `confidence_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_temp095_post4` | `confidence_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_temp095_post4` | `confidence_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_temp095_post4` | `confidence_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_temp095_post4` | `confidence_weighting` | prospective_e | 223 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_temp095_post4` | `confidence_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round10_conf_temp095_post4` | `confidence_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_temp095_post4` | `confidence_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_door3_post5` | `replay_cap_bridge_revisit` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_door3_post5` | `replay_cap_bridge_revisit` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_door3_post5` | `replay_cap_bridge_revisit` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_door3_post5` | `replay_cap_bridge_revisit` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_door3_post5` | `replay_cap_bridge_revisit` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_door3_post5` | `replay_cap_bridge_revisit` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_door3_post5` | `replay_cap_bridge_revisit` | prospective_e | 223 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_door3_post5` | `replay_cap_bridge_revisit` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round10_door3_post5` | `replay_cap_bridge_revisit` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_door3_post5` | `replay_cap_bridge_revisit` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_c | 181 | 0.4844 | -0.5156 | -0.5156 | -0.5156 |
| `round10_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_c | 191 | 0.2812 | -0.7188 | -0.7188 | -0.7188 |
| `round10_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_d | 197 | 0.1719 | -0.8281 | -0.8281 | -0.8281 |
| `round10_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_d | 199 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_d | 211 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_e | 223 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_f | 233 | 0.0000 | -1.0000 | -0.6250 | -1.0000 |
| `round10_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_f | 239 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_f | 241 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_stoch25_last_two` | `rsm_stochastic_credit_horizon` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_stoch25_last_two` | `rsm_stochastic_credit_horizon` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_stoch25_last_two` | `rsm_stochastic_credit_horizon` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_stoch25_last_two` | `rsm_stochastic_credit_horizon` | prospective_d | 197 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_stoch25_last_two` | `rsm_stochastic_credit_horizon` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_stoch25_last_two` | `rsm_stochastic_credit_horizon` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_stoch25_last_two` | `rsm_stochastic_credit_horizon` | prospective_e | 223 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_stoch25_last_two` | `rsm_stochastic_credit_horizon` | prospective_f | 233 | 0.4531 | -0.5469 | -0.1719 | -0.5469 |
| `round10_stoch25_last_two` | `rsm_stochastic_credit_horizon` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_stoch25_last_two` | `rsm_stochastic_credit_horizon` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_d | 197 | 0.4375 | -0.5625 | -0.5625 | -0.5625 |
| `round10_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_d | 199 | 0.4844 | -0.5156 | -0.5156 | -0.5156 |
| `round10_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_d | 211 | 0.5938 | -0.4062 | -0.4062 | -0.4062 |
| `round10_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_e | 223 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_f | 233 | 0.6250 | -0.3750 | 0.0000 | -0.3750 |
| `round10_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_stoch75_last_two` | `rsm_stochastic_credit_horizon` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_stoch75_last_two` | `rsm_stochastic_credit_horizon` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_stoch75_last_two` | `rsm_stochastic_credit_horizon` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_stoch75_last_two` | `rsm_stochastic_credit_horizon` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_stoch75_last_two` | `rsm_stochastic_credit_horizon` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_stoch75_last_two` | `rsm_stochastic_credit_horizon` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_stoch75_last_two` | `rsm_stochastic_credit_horizon` | prospective_e | 223 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_stoch75_last_two` | `rsm_stochastic_credit_horizon` | prospective_f | 233 | 0.2031 | -0.7969 | -0.4219 | -0.7969 |
| `round10_stoch75_last_two` | `rsm_stochastic_credit_horizon` | prospective_f | 239 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_stoch75_last_two` | `rsm_stochastic_credit_horizon` | prospective_f | 241 | 0.5312 | -0.4688 | -0.4688 | -0.4688 |
| `round10_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_d | 197 | 0.4375 | -0.5625 | -0.5625 | -0.5625 |
| `round10_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_e | 223 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round10_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_warm2_last_step` | `rsm_detached_warmup_final_step` | prospective_c | 181 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_warm2_last_step` | `rsm_detached_warmup_final_step` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_warm2_last_step` | `rsm_detached_warmup_final_step` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_warm2_last_step` | `rsm_detached_warmup_final_step` | prospective_d | 197 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_warm2_last_step` | `rsm_detached_warmup_final_step` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_warm2_last_step` | `rsm_detached_warmup_final_step` | prospective_d | 211 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_warm2_last_step` | `rsm_detached_warmup_final_step` | prospective_e | 223 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_warm2_last_step` | `rsm_detached_warmup_final_step` | prospective_f | 233 | 0.4531 | -0.5469 | -0.1719 | -0.5469 |
| `round10_warm2_last_step` | `rsm_detached_warmup_final_step` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_warm2_last_step` | `rsm_detached_warmup_final_step` | prospective_f | 241 | 0.9531 | -0.0469 | -0.0469 | -0.0469 |
| `round12_conf_post4` | `confidence_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post4` | `confidence_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post4` | `confidence_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post4` | `confidence_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post4` | `confidence_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post4` | `confidence_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post4` | `confidence_weighting` | prospective_e | 223 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post4` | `confidence_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round12_conf_post4` | `confidence_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post4` | `confidence_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_temp095_post4` | `confidence_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_temp095_post4` | `confidence_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_temp095_post4` | `confidence_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_temp095_post4` | `confidence_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_temp095_post4` | `confidence_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_temp095_post4` | `confidence_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_temp095_post4` | `confidence_weighting` | prospective_e | 223 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_temp095_post4` | `confidence_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round12_conf_temp095_post4` | `confidence_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_temp095_post4` | `confidence_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_d | 197 | 0.4375 | -0.5625 | -0.5625 | -0.5625 |
| `round12_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_d | 199 | 0.4844 | -0.5156 | -0.5156 | -0.5156 |
| `round12_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_d | 211 | 0.5938 | -0.4062 | -0.4062 | -0.4062 |
| `round12_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_e | 223 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_f | 233 | 0.6250 | -0.3750 | 0.0000 | -0.3750 |
| `round12_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_e | 223 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round12_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_warm2_last_step` | `rsm_detached_warmup_final_step` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_warm2_last_step` | `rsm_detached_warmup_final_step` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_warm2_last_step` | `rsm_detached_warmup_final_step` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_warm2_last_step` | `rsm_detached_warmup_final_step` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_warm2_last_step` | `rsm_detached_warmup_final_step` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_warm2_last_step` | `rsm_detached_warmup_final_step` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_warm2_last_step` | `rsm_detached_warmup_final_step` | prospective_e | 223 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_warm2_last_step` | `rsm_detached_warmup_final_step` | prospective_f | 233 | 0.4375 | -0.5625 | -0.1875 | -0.5625 |
| `round12_warm2_last_step` | `rsm_detached_warmup_final_step` | prospective_f | 239 | 0.6250 | -0.3750 | -0.3750 | -0.3750 |
| `round12_warm2_last_step` | `rsm_detached_warmup_final_step` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post4` | `confidence_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post4` | `confidence_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post4` | `confidence_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post4` | `confidence_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post4` | `confidence_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post4` | `confidence_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post4` | `confidence_weighting` | prospective_e | 223 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post4` | `confidence_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round7_conf_post4` | `confidence_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post4` | `confidence_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_c | 181 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_c | 191 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_d | 197 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_d | 199 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_d | 211 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_e | 223 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_f | 233 | 0.1094 | -0.8906 | -0.5156 | -0.8906 |
| `round7_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_f | 239 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_f | 241 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_c | 181 | 0.5156 | -0.4844 | -0.4844 | -0.4844 |
| `round7_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_d | 197 | 0.4375 | -0.5625 | -0.5625 | -0.5625 |
| `round7_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_d | 199 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_e | 223 | 0.5312 | -0.4688 | -0.4688 | -0.4688 |
| `round7_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_f | 233 | 0.0000 | -1.0000 | -0.6250 | -1.0000 |
| `round7_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_f | 239 | 0.5156 | -0.4844 | -0.4844 | -0.4844 |
| `round7_stoch50_last_two` | `rsm_stochastic_credit_horizon` | prospective_f | 241 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_c | 181 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_d | 197 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_d | 199 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_d | 211 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_e | 223 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_f | 233 | 0.4531 | -0.5469 | -0.1719 | -0.5469 |
| `round7_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_f | 239 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_warm1_last_step` | `rsm_detached_warmup_final_step` | prospective_f | 241 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `settling_round10_last_step` | `settling_diagnostics` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_last_step` | `settling_diagnostics` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_last_step` | `settling_diagnostics` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_last_step` | `settling_diagnostics` | prospective_d | 197 | 0.4375 | -0.5625 | -0.5625 | -0.5625 |
| `settling_round10_last_step` | `settling_diagnostics` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_last_step` | `settling_diagnostics` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_last_step` | `settling_diagnostics` | prospective_e | 223 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_last_step` | `settling_diagnostics` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `settling_round10_last_step` | `settling_diagnostics` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_last_step` | `settling_diagnostics` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_last_two` | `settling_diagnostics` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_last_two` | `settling_diagnostics` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_last_two` | `settling_diagnostics` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_last_two` | `settling_diagnostics` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_last_two` | `settling_diagnostics` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_last_two` | `settling_diagnostics` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_last_two` | `settling_diagnostics` | prospective_e | 223 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_last_two` | `settling_diagnostics` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `settling_round10_last_two` | `settling_diagnostics` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_last_two` | `settling_diagnostics` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_stoch50` | `settling_diagnostics` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_stoch50` | `settling_diagnostics` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_stoch50` | `settling_diagnostics` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_stoch50` | `settling_diagnostics` | prospective_d | 197 | 0.4375 | -0.5625 | -0.5625 | -0.5625 |
| `settling_round10_stoch50` | `settling_diagnostics` | prospective_d | 199 | 0.4844 | -0.5156 | -0.5156 | -0.5156 |
| `settling_round10_stoch50` | `settling_diagnostics` | prospective_d | 211 | 0.5938 | -0.4062 | -0.4062 | -0.4062 |
| `settling_round10_stoch50` | `settling_diagnostics` | prospective_e | 223 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_stoch50` | `settling_diagnostics` | prospective_f | 233 | 0.6250 | -0.3750 | 0.0000 | -0.3750 |
| `settling_round10_stoch50` | `settling_diagnostics` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round10_stoch50` | `settling_diagnostics` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round7_last_step` | `settling_diagnostics` | prospective_c | 181 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `settling_round7_last_step` | `settling_diagnostics` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round7_last_step` | `settling_diagnostics` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `settling_round7_last_step` | `settling_diagnostics` | prospective_d | 197 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `settling_round7_last_step` | `settling_diagnostics` | prospective_d | 199 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `settling_round7_last_step` | `settling_diagnostics` | prospective_d | 211 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `settling_round7_last_step` | `settling_diagnostics` | prospective_e | 223 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `settling_round7_last_step` | `settling_diagnostics` | prospective_f | 233 | 0.4531 | -0.5469 | -0.1719 | -0.5469 |
| `settling_round7_last_step` | `settling_diagnostics` | prospective_f | 239 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `settling_round7_last_step` | `settling_diagnostics` | prospective_f | 241 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
