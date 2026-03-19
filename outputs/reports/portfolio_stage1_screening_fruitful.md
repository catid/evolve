# Portfolio Stage 1 Fruitful Screening

- track budget: `20` candidates
- advancing challengers from this track: `['round10', 'round10_post_unlock_x4_dis025', 'round10_post_unlock_x5']`
- incumbent round6 dev SARE/token/single: `0.8889` / `0.8472` / `0.8889`
- family counts: `{'near_neighbor_rounds': 6, 'hard_mixed_weighting': 6, 'hard_postunlock_weighting': 2, 'hard_smoothing_postunlock': 3, 'near_neighbor_temperature': 3}`

| Candidate | Family | Direction | Dev Mean | Δ vs round6 | Candidate-token | Candidate-single | Failures | Observed Specs | Stage 1 | Reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `round10` | `near_neighbor_rounds` | `late_round_selection_late` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `advance` |
| `round10_post_unlock_x4_dis025` | `hard_mixed_weighting` | `late_round_mixed_post_unlock_disagreement` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `advance` |
| `round10_post_unlock_x5` | `hard_postunlock_weighting` | `late_round_post_unlock_weight` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `advance` |
| `round10_post_unlock_x5_dis025` | `hard_mixed_weighting` | `late_round_mixed_post_unlock_disagreement` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside fruitful top-3` |
| `round10_post_unlock_x5_dis050` | `hard_mixed_weighting` | `late_round_mixed_post_unlock_disagreement` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside fruitful top-3` |
| `round10_temp095_post5` | `hard_smoothing_postunlock` | `mild_teacher_smoothing_plus_post_unlock` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside fruitful top-3` |
| `round10_temp_090` | `near_neighbor_temperature` | `late_round_teacher_temperature_090` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside fruitful top-3` |
| `round12` | `near_neighbor_rounds` | `late_round_selection_late` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside fruitful top-3` |
| `round12_post_unlock_x4_dis025` | `hard_mixed_weighting` | `late_round_mixed_post_unlock_disagreement` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside fruitful top-3` |
| `round12_post_unlock_x5` | `hard_postunlock_weighting` | `late_round_post_unlock_weight` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside fruitful top-3` |
| `round12_post_unlock_x5_dis025` | `hard_mixed_weighting` | `late_round_mixed_post_unlock_disagreement` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside fruitful top-3` |
| `round12_post_unlock_x5_dis050` | `hard_mixed_weighting` | `late_round_mixed_post_unlock_disagreement` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside fruitful top-3` |
| `round12_temp095_post5` | `hard_smoothing_postunlock` | `mild_teacher_smoothing_plus_post_unlock` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside fruitful top-3` |
| `round12_temp_090` | `near_neighbor_temperature` | `late_round_teacher_temperature_090` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside fruitful top-3` |
| `round7` | `near_neighbor_rounds` | `late_round_selection_plus1` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside fruitful top-3` |
| `round7_temp095_post5` | `hard_smoothing_postunlock` | `mild_teacher_smoothing_plus_post_unlock` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside fruitful top-3` |
| `round7_temp_090` | `near_neighbor_temperature` | `late_round_teacher_temperature_090` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside fruitful top-3` |
| `round8` | `near_neighbor_rounds` | `late_round_selection_mid` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside fruitful top-3` |
| `round9` | `near_neighbor_rounds` | `late_round_selection_mid` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `1` | `9` / `9` | `pass` | `stop: outside fruitful top-3` |
| `round5` | `near_neighbor_rounds` | `round_selection_round5` | `0.8368` | `-0.0521` | `-0.0104` | `-0.0521` | `1` | `9` / `9` | `stop` | `stop: below incumbent dev mean` |

| Candidate | Family | Block | Seed | Greedy Success | Δ vs round6 | Δ vs token_dense | Δ vs single_expert |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `round10` | `near_neighbor_rounds` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10` | `near_neighbor_rounds` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10` | `near_neighbor_rounds` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10` | `near_neighbor_rounds` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10` | `near_neighbor_rounds` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10` | `near_neighbor_rounds` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10` | `near_neighbor_rounds` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round10` | `near_neighbor_rounds` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10` | `near_neighbor_rounds` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x4_dis025` | `hard_mixed_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x4_dis025` | `hard_mixed_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x4_dis025` | `hard_mixed_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x4_dis025` | `hard_mixed_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x4_dis025` | `hard_mixed_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x4_dis025` | `hard_mixed_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x4_dis025` | `hard_mixed_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round10_post_unlock_x4_dis025` | `hard_mixed_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x4_dis025` | `hard_mixed_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5` | `hard_postunlock_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5` | `hard_postunlock_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5` | `hard_postunlock_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5` | `hard_postunlock_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5` | `hard_postunlock_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5` | `hard_postunlock_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5` | `hard_postunlock_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round10_post_unlock_x5` | `hard_postunlock_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5` | `hard_postunlock_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5_dis025` | `hard_mixed_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5_dis025` | `hard_mixed_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5_dis025` | `hard_mixed_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5_dis025` | `hard_mixed_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5_dis025` | `hard_mixed_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5_dis025` | `hard_mixed_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5_dis025` | `hard_mixed_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round10_post_unlock_x5_dis025` | `hard_mixed_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5_dis025` | `hard_mixed_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5_dis050` | `hard_mixed_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5_dis050` | `hard_mixed_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5_dis050` | `hard_mixed_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5_dis050` | `hard_mixed_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5_dis050` | `hard_mixed_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5_dis050` | `hard_mixed_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5_dis050` | `hard_mixed_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round10_post_unlock_x5_dis050` | `hard_mixed_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_post_unlock_x5_dis050` | `hard_mixed_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp095_post5` | `hard_smoothing_postunlock` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp095_post5` | `hard_smoothing_postunlock` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp095_post5` | `hard_smoothing_postunlock` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp095_post5` | `hard_smoothing_postunlock` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp095_post5` | `hard_smoothing_postunlock` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp095_post5` | `hard_smoothing_postunlock` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp095_post5` | `hard_smoothing_postunlock` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round10_temp095_post5` | `hard_smoothing_postunlock` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp095_post5` | `hard_smoothing_postunlock` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp_090` | `near_neighbor_temperature` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp_090` | `near_neighbor_temperature` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp_090` | `near_neighbor_temperature` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp_090` | `near_neighbor_temperature` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp_090` | `near_neighbor_temperature` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp_090` | `near_neighbor_temperature` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp_090` | `near_neighbor_temperature` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round10_temp_090` | `near_neighbor_temperature` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round10_temp_090` | `near_neighbor_temperature` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12` | `near_neighbor_rounds` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12` | `near_neighbor_rounds` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12` | `near_neighbor_rounds` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12` | `near_neighbor_rounds` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12` | `near_neighbor_rounds` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12` | `near_neighbor_rounds` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12` | `near_neighbor_rounds` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round12` | `near_neighbor_rounds` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12` | `near_neighbor_rounds` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x4_dis025` | `hard_mixed_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x4_dis025` | `hard_mixed_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x4_dis025` | `hard_mixed_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x4_dis025` | `hard_mixed_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x4_dis025` | `hard_mixed_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x4_dis025` | `hard_mixed_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x4_dis025` | `hard_mixed_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round12_post_unlock_x4_dis025` | `hard_mixed_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x4_dis025` | `hard_mixed_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5` | `hard_postunlock_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5` | `hard_postunlock_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5` | `hard_postunlock_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5` | `hard_postunlock_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5` | `hard_postunlock_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5` | `hard_postunlock_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5` | `hard_postunlock_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round12_post_unlock_x5` | `hard_postunlock_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5` | `hard_postunlock_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5_dis025` | `hard_mixed_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5_dis025` | `hard_mixed_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5_dis025` | `hard_mixed_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5_dis025` | `hard_mixed_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5_dis025` | `hard_mixed_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5_dis025` | `hard_mixed_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5_dis025` | `hard_mixed_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round12_post_unlock_x5_dis025` | `hard_mixed_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5_dis025` | `hard_mixed_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5_dis050` | `hard_mixed_weighting` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5_dis050` | `hard_mixed_weighting` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5_dis050` | `hard_mixed_weighting` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5_dis050` | `hard_mixed_weighting` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5_dis050` | `hard_mixed_weighting` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5_dis050` | `hard_mixed_weighting` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5_dis050` | `hard_mixed_weighting` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round12_post_unlock_x5_dis050` | `hard_mixed_weighting` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_post_unlock_x5_dis050` | `hard_mixed_weighting` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp095_post5` | `hard_smoothing_postunlock` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp095_post5` | `hard_smoothing_postunlock` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp095_post5` | `hard_smoothing_postunlock` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp095_post5` | `hard_smoothing_postunlock` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp095_post5` | `hard_smoothing_postunlock` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp095_post5` | `hard_smoothing_postunlock` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp095_post5` | `hard_smoothing_postunlock` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round12_temp095_post5` | `hard_smoothing_postunlock` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp095_post5` | `hard_smoothing_postunlock` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp_090` | `near_neighbor_temperature` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp_090` | `near_neighbor_temperature` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp_090` | `near_neighbor_temperature` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp_090` | `near_neighbor_temperature` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp_090` | `near_neighbor_temperature` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp_090` | `near_neighbor_temperature` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp_090` | `near_neighbor_temperature` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round12_temp_090` | `near_neighbor_temperature` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round12_temp_090` | `near_neighbor_temperature` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round5` | `near_neighbor_rounds` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round5` | `near_neighbor_rounds` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round5` | `near_neighbor_rounds` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round5` | `near_neighbor_rounds` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round5` | `near_neighbor_rounds` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round5` | `near_neighbor_rounds` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round5` | `near_neighbor_rounds` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round5` | `near_neighbor_rounds` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round5` | `near_neighbor_rounds` | prospective_f | 241 | 0.5312 | -0.4688 | -0.4688 | -0.4688 |
| `round7` | `near_neighbor_rounds` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7` | `near_neighbor_rounds` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7` | `near_neighbor_rounds` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7` | `near_neighbor_rounds` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7` | `near_neighbor_rounds` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7` | `near_neighbor_rounds` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7` | `near_neighbor_rounds` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round7` | `near_neighbor_rounds` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7` | `near_neighbor_rounds` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp095_post5` | `hard_smoothing_postunlock` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp095_post5` | `hard_smoothing_postunlock` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp095_post5` | `hard_smoothing_postunlock` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp095_post5` | `hard_smoothing_postunlock` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp095_post5` | `hard_smoothing_postunlock` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp095_post5` | `hard_smoothing_postunlock` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp095_post5` | `hard_smoothing_postunlock` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round7_temp095_post5` | `hard_smoothing_postunlock` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp095_post5` | `hard_smoothing_postunlock` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp_090` | `near_neighbor_temperature` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp_090` | `near_neighbor_temperature` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp_090` | `near_neighbor_temperature` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp_090` | `near_neighbor_temperature` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp_090` | `near_neighbor_temperature` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp_090` | `near_neighbor_temperature` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp_090` | `near_neighbor_temperature` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round7_temp_090` | `near_neighbor_temperature` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round7_temp_090` | `near_neighbor_temperature` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round8` | `near_neighbor_rounds` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round8` | `near_neighbor_rounds` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round8` | `near_neighbor_rounds` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round8` | `near_neighbor_rounds` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round8` | `near_neighbor_rounds` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round8` | `near_neighbor_rounds` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round8` | `near_neighbor_rounds` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round8` | `near_neighbor_rounds` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round8` | `near_neighbor_rounds` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round9` | `near_neighbor_rounds` | prospective_c | 181 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round9` | `near_neighbor_rounds` | prospective_c | 191 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round9` | `near_neighbor_rounds` | prospective_c | 193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round9` | `near_neighbor_rounds` | prospective_d | 197 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round9` | `near_neighbor_rounds` | prospective_d | 199 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round9` | `near_neighbor_rounds` | prospective_d | 211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round9` | `near_neighbor_rounds` | prospective_f | 233 | 1.0000 | 0.0000 | 0.3750 | 0.0000 |
| `round9` | `near_neighbor_rounds` | prospective_f | 239 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `round9` | `near_neighbor_rounds` | prospective_f | 241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
