# Memory Next Stage D1 Verification

- exploit survivors entering verification: `8`
- exploration survivors entering verification: `0`

## Exploit Rerun Consolidation

| Candidate | Family | Screen Success | Rerun Success | Rerun Greedy |
| --- | --- | --- | --- | --- |
| partial225_gap_t055 | actor_hidden_continuation | 0.5208 | 0.5208 | 0.5208 |
| partial225_greedy | actor_hidden_continuation | 0.5208 | 0.5208 | 0.5208 |
| select_partial225_greedy | checkpoint_selection | 0.5208 | 0.5208 | 0.5208 |
| select_partial22_greedy | checkpoint_selection | 0.5208 | 0.5208 | 0.5208 |
| partial22_conf_gate085_t055 | sampled_to_greedy_decode | 0.5208 | 0.5208 | 0.5208 |
| partial22_conf_gate092_t055 | sampled_to_greedy_decode | 0.5208 | 0.5208 | 0.5208 |
| partial22_cons2_t05 | small_branch_consensus | 0.5208 | 0.5208 | 0.5208 |
| partial22_cons2_t055 | small_branch_consensus | 0.5208 | 0.5208 | 0.5208 |

## Interpretation

- Stage D1 consolidates the fresh-root reruns that were already required by the family fair-shot rule.
- Only directionally consistent rerun lines are allowed to enter holdout and anti-regression work.
