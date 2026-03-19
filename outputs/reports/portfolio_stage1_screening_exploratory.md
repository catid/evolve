# Portfolio Stage 1 Exploratory Screening

- track budget: `20` candidates
- advancing challengers from this track: `['round10_carry2_post4', 'round10_conf_post4', 'round10_door2_post4']`
- incumbent round6 dev SARE/token/single: `0.8889` / `0.8472` / `0.8889`
- family counts: `{'replay_cap_bridge_revisit': 5, 'bridge_weighting': 6, 'confidence_weighting': 6, 'teacher_smoothing': 3}`

| Candidate | Family | Direction | Dev Mean | Δ vs round6 | Candidate-token | Candidate-single | Failures | Observed Specs | Stage 1 | Reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `round10_carry2_post4` | `bridge_weighting` | `late_round_carry_key_bridge` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `advance` |
| `round10_conf_post4` | `confidence_weighting` | `late_round_teacher_confidence_weighting` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `advance` |
| `round10_door2_post4` | `bridge_weighting` | `late_round_locked_door_bridge` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `advance` |
| `round10_temp_095` | `teacher_smoothing` | `mild_teacher_smoothing` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside exploratory top-3` |
| `round12_carry2_post4` | `bridge_weighting` | `late_round_carry_key_bridge` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside exploratory top-3` |
| `round12_conf_post4` | `confidence_weighting` | `late_round_teacher_confidence_weighting` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside exploratory top-3` |
| `round12_conf_post5` | `confidence_weighting` | `late_round_teacher_confidence_weighting` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside exploratory top-3` |
| `round12_door2_post4` | `bridge_weighting` | `late_round_locked_door_bridge` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside exploratory top-3` |
| `round12_temp_095` | `teacher_smoothing` | `mild_teacher_smoothing` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside exploratory top-3` |
| `round7_carry2_post4` | `bridge_weighting` | `late_round_carry_key_bridge` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside exploratory top-3` |
| `round7_conf_post4` | `confidence_weighting` | `late_round_teacher_confidence_weighting` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside exploratory top-3` |
| `round7_conf_post5` | `confidence_weighting` | `late_round_teacher_confidence_weighting` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside exploratory top-3` |
| `round7_door2_post4` | `bridge_weighting` | `late_round_locked_door_bridge` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside exploratory top-3` |
| `round7_temp_095` | `teacher_smoothing` | `mild_teacher_smoothing` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside exploratory top-3` |
| `round10_conf_post5` | `confidence_weighting` | `late_round_teacher_confidence_weighting` | `0.7778` | `-0.1111` | `-0.0694` | `-0.1111` | `2` | `9` / `9` | `stop` | `stop: new complete-seed failures` |
| `round12_phase_balanced_4096` | `replay_cap_bridge_revisit` | `late_round_phase_balanced_replay` | `0.4566` | `-0.4323` | `-0.3906` | `-0.4323` | `4` | `9` / `9` | `stop` | `stop: new complete-seed failures` |
| `round10_cap_balanced_4096` | `replay_cap_bridge_revisit` | `cap_balanced_recent_replay` | `0.2257` | `-0.6632` | `-0.6215` | `-0.6632` | `6` | `9` / `9` | `stop` | `stop: new complete-seed failures` |
| `round10_phase_balanced_4096` | `replay_cap_bridge_revisit` | `late_round_phase_balanced_replay` | `0.1042` | `-0.7847` | `-0.7431` | `-0.7847` | `6` | `9` / `9` | `stop` | `stop: new complete-seed failures` |
| `round7_phase_balanced_4096` | `replay_cap_bridge_revisit` | `late_round_phase_balanced_replay` | `0.0122` | `-0.8767` | `-0.8351` | `-0.8767` | `8` | `9` / `9` | `stop` | `stop: new complete-seed failures` |
| `round10_cap_recent_4096` | `replay_cap_bridge_revisit` | `cap_recent_replay` | `0.0000` | `-0.8889` | `-0.8472` | `-0.8889` | `4` | `4` / `9` | `stop` | `stop: catastrophic family prune after incomplete calibration` |

| Candidate | Family | Block | Seed | Greedy Success | Δ vs round6 | Δ vs token_dense | Δ vs single_expert |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `round10_cap_balanced_4096` | `replay_cap_bridge_revisit` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_cap_balanced_4096` | `replay_cap_bridge_revisit` | prospective_c | 191 | 0.2812 | -0.7188 | -0.7188 | -0.7188 |
| `round10_cap_balanced_4096` | `replay_cap_bridge_revisit` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_cap_balanced_4096` | `replay_cap_bridge_revisit` | prospective_d | 197 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_cap_balanced_4096` | `replay_cap_bridge_revisit` | prospective_d | 199 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_cap_balanced_4096` | `replay_cap_bridge_revisit` | prospective_d | 211 | 0.7500 | -0.2500 | -0.2500 | -0.2500 |
| `round10_cap_balanced_4096` | `replay_cap_bridge_revisit` | prospective_f | 233 | 0.0000 | -1.0000 | -0.6250 | -1.0000 |
| `round10_cap_balanced_4096` | `replay_cap_bridge_revisit` | prospective_f | 239 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_cap_balanced_4096` | `replay_cap_bridge_revisit` | prospective_f | 241 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_cap_recent_4096` | `replay_cap_bridge_revisit` | prospective_d | 197 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_cap_recent_4096` | `replay_cap_bridge_revisit` | prospective_d | 199 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_cap_recent_4096` | `replay_cap_bridge_revisit` | prospective_d | 211 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_cap_recent_4096` | `replay_cap_bridge_revisit` | prospective_f | 233 | 0.0000 | -1.0000 | -0.6250 | -1.0000 |
| `round10_carry2_post4` | `bridge_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_carry2_post4` | `bridge_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_carry2_post4` | `bridge_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_carry2_post4` | `bridge_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_carry2_post4` | `bridge_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_carry2_post4` | `bridge_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_carry2_post4` | `bridge_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round10_carry2_post4` | `bridge_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_carry2_post4` | `bridge_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post4` | `confidence_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post4` | `confidence_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post4` | `confidence_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post4` | `confidence_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post4` | `confidence_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post4` | `confidence_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post4` | `confidence_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round10_conf_post4` | `confidence_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post4` | `confidence_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post5` | `confidence_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post5` | `confidence_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post5` | `confidence_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post5` | `confidence_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post5` | `confidence_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post5` | `confidence_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_conf_post5` | `confidence_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round10_conf_post5` | `confidence_weighting` | prospective_f | 239 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_conf_post5` | `confidence_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_door2_post4` | `bridge_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_door2_post4` | `bridge_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_door2_post4` | `bridge_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_door2_post4` | `bridge_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_door2_post4` | `bridge_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_door2_post4` | `bridge_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_door2_post4` | `bridge_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round10_door2_post4` | `bridge_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_door2_post4` | `bridge_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_c | 181 | 0.4844 | -0.5156 | -0.5156 | -0.5156 |
| `round10_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_c | 191 | 0.2812 | -0.7188 | -0.7188 | -0.7188 |
| `round10_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_d | 197 | 0.1719 | -0.8281 | -0.8281 | -0.8281 |
| `round10_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_d | 199 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_d | 211 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_f | 233 | 0.0000 | -1.0000 | -0.6250 | -1.0000 |
| `round10_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_f | 239 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_f | 241 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round10_temp_095` | `teacher_smoothing` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp_095` | `teacher_smoothing` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp_095` | `teacher_smoothing` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp_095` | `teacher_smoothing` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp_095` | `teacher_smoothing` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp_095` | `teacher_smoothing` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp_095` | `teacher_smoothing` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round10_temp_095` | `teacher_smoothing` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp_095` | `teacher_smoothing` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_carry2_post4` | `bridge_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_carry2_post4` | `bridge_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_carry2_post4` | `bridge_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_carry2_post4` | `bridge_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_carry2_post4` | `bridge_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_carry2_post4` | `bridge_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_carry2_post4` | `bridge_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round12_carry2_post4` | `bridge_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_carry2_post4` | `bridge_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post4` | `confidence_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post4` | `confidence_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post4` | `confidence_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post4` | `confidence_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post4` | `confidence_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post4` | `confidence_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post4` | `confidence_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round12_conf_post4` | `confidence_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post4` | `confidence_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post5` | `confidence_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post5` | `confidence_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post5` | `confidence_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post5` | `confidence_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post5` | `confidence_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post5` | `confidence_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post5` | `confidence_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round12_conf_post5` | `confidence_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_conf_post5` | `confidence_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_door2_post4` | `bridge_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_door2_post4` | `bridge_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_door2_post4` | `bridge_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_door2_post4` | `bridge_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_door2_post4` | `bridge_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_door2_post4` | `bridge_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_door2_post4` | `bridge_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round12_door2_post4` | `bridge_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_door2_post4` | `bridge_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_c | 181 | 0.4844 | -0.5156 | -0.5156 | -0.5156 |
| `round12_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_d | 197 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round12_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_d | 199 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round12_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round12_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_f | 239 | 0.6250 | -0.3750 | -0.3750 | -0.3750 |
| `round12_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_f | 241 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round12_temp_095` | `teacher_smoothing` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp_095` | `teacher_smoothing` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp_095` | `teacher_smoothing` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp_095` | `teacher_smoothing` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp_095` | `teacher_smoothing` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp_095` | `teacher_smoothing` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp_095` | `teacher_smoothing` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round12_temp_095` | `teacher_smoothing` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp_095` | `teacher_smoothing` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_carry2_post4` | `bridge_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_carry2_post4` | `bridge_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_carry2_post4` | `bridge_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_carry2_post4` | `bridge_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_carry2_post4` | `bridge_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_carry2_post4` | `bridge_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_carry2_post4` | `bridge_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round7_carry2_post4` | `bridge_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_carry2_post4` | `bridge_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post4` | `confidence_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post4` | `confidence_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post4` | `confidence_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post4` | `confidence_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post4` | `confidence_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post4` | `confidence_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post4` | `confidence_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round7_conf_post4` | `confidence_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post4` | `confidence_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post5` | `confidence_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post5` | `confidence_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post5` | `confidence_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post5` | `confidence_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post5` | `confidence_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post5` | `confidence_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post5` | `confidence_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round7_conf_post5` | `confidence_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_conf_post5` | `confidence_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_door2_post4` | `bridge_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_door2_post4` | `bridge_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_door2_post4` | `bridge_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_door2_post4` | `bridge_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_door2_post4` | `bridge_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_door2_post4` | `bridge_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_door2_post4` | `bridge_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round7_door2_post4` | `bridge_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_door2_post4` | `bridge_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_c | 181 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_c | 191 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_d | 197 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_d | 199 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_d | 211 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_f | 233 | 0.1094 | -0.8906 | -0.5156 | -0.8906 |
| `round7_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_f | 239 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_phase_balanced_4096` | `replay_cap_bridge_revisit` | prospective_f | 241 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| `round7_temp_095` | `teacher_smoothing` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp_095` | `teacher_smoothing` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp_095` | `teacher_smoothing` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp_095` | `teacher_smoothing` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp_095` | `teacher_smoothing` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp_095` | `teacher_smoothing` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp_095` | `teacher_smoothing` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round7_temp_095` | `teacher_smoothing` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp_095` | `teacher_smoothing` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
