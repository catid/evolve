# Memory Next Stage D2 Fairness Holdout

- matched holdout runs: `10`
- stage-d2 survivors: `7`
- holdout POR gap anchor: `0.4792`
- holdout GRU gap anchor: `0.4479`
- holdout actor-hidden greedy anchor: `0.5208`

| Candidate | Family | Greedy | Candidate | Lower | Gap | Shoulder | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| select_gru_long11_t055 | checkpoint_selection | 0.0000 | 0.4479 | 0.3438 | 0.4479 | 0.5104 | control |
| select_por_switchy7_t055 | checkpoint_selection | 0.0000 | 0.4792 | 0.3750 | 0.4792 | 0.5208 | control |
| partial225_gap_t055 | actor_hidden_continuation | 0.5208 | 0.5208 | 0.5208 | 0.5208 | 0.5208 | advance |
| partial225_greedy | actor_hidden_continuation | 0.5208 | 0.5208 | 0.5208 | 0.5208 | 0.5208 | advance |
| select_partial225_greedy | checkpoint_selection | 0.5208 | 0.5208 | 0.5208 | 0.5208 | 0.5208 | advance |
| select_partial22_greedy | checkpoint_selection | 0.5208 | 0.5208 | 0.5208 | 0.5208 | 0.5208 | control |
| partial22_conf_gate085_t055 | sampled_to_greedy_decode | 0.5208 | 0.5208 | 0.5208 | 0.5208 | 0.5208 | advance |
| partial22_conf_gate092_t055 | sampled_to_greedy_decode | 0.5208 | 0.5208 | 0.5208 | 0.5208 | 0.5208 | advance |
| partial22_cons2_t05 | small_branch_consensus | 0.5208 | 0.5208 | 0.5208 | 0.5208 | 0.5208 | advance |
| partial22_cons2_t055 | small_branch_consensus | 0.5208 | 0.5208 | 0.5208 | 0.5208 | 0.5208 | advance |

## Interpretation

- Stage D2 asks whether a candidate still looks real on Memory holdout under matched POR, recurrent-control, and actor-hidden references.
- Surviving candidates must stay meaningful in the strongest-gap band without collapsing on greedy or outside a single narrow point.
