# Current Summary

- `flat_dense` remains the strongest greedy DoorKey control.
- recovered `token_dense` with `ppo.ent_coef=0.001` remains the canonical tokenized control.
- PPO-only `SARE` is still negative on greedy DoorKey.
- The current reopened routed result is narrower: teacher-logit KL learner-state supervision for `SARE` now passes the external 3-seed `64`-episode gate on DoorKey. See [lss_robustness_multiseed_report.md](outputs/reports/lss_robustness_multiseed_report.md).
- The winning bounded method is `kl` teacher targets with `append_all` aggregation; capped replay variants were negative. See [lss_robustness_sweep_report.md](outputs/reports/lss_robustness_sweep_report.md).
- On the revived seed `19`, routing remains meaningful rather than collapsing into a dense shortcut. See [lss_route_integrity_report.md](outputs/reports/lss_route_integrity_report.md).
- The current repo recommendation is to continue routed work on DoorKey only under this teacher-guided extraction claim, not as a PPO-alone routed win. See [lss_robustness_decision_memo.md](outputs/reports/lss_robustness_decision_memo.md).
