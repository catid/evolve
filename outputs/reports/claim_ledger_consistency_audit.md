# Claim Ledger Consistency Audit

| Ledger Family | Ledger Status | Historical Packs | Replay Verdicts | Classification | Detail |
| --- | --- | --- | --- | --- | --- |
| PPO tuning / checkpoint dynamics | `negative` | `checkpoint_dynamics` | `FAIL: claim remains frozen` | `consistent` | negative row replays to a hard gate failure |
| Entropy schedules | `negative` | `entropy_schedule` | `FAIL: claim remains frozen` | `consistent` | negative row replays to a hard gate failure |
| Self-imitation | `negative` | `self_imitation` | `INCONCLUSIVE: missing prerequisites` | `consistent but narrower under current gate` | negative row stays blocked, but current gate treats it as not fully comparable rather than directly claim-failing |
| Offline teacher distillation | `negative` | `policy_distillation` | `INCONCLUSIVE: missing prerequisites` | `consistent but narrower under current gate` | negative row stays blocked, but current gate treats it as not fully comparable rather than directly claim-failing |
| Learner-state supervision | `bounded positive` | `learner_state_supervision_initial`, `teacher_extraction_multiseed`, `lss_robustness_multiseed` | `INCONCLUSIVE: missing prerequisites`, `INCONCLUSIVE: missing prerequisites`, `INCONCLUSIVE: missing prerequisites` | `consistent but narrower under current gate` | historical positive signal still does not clear the modern frozen gate |
| Claim hardening | `bounded positive` | `claim_hardening` | `INCONCLUSIVE: missing prerequisites` | `consistent but narrower under current gate` | historical positive signal still does not clear the modern frozen gate |
| Claim broadening | `bounded positive` | `claim_broadening` | `FAIL: claim remains frozen` | `consistent but narrower under current gate` | historical positive signal still does not clear the modern frozen gate |
| Multi-expert hardening | `bounded positive` | `multi_expert_hardening` | `INCONCLUSIVE: missing prerequisites` | `consistent but narrower under current gate` | historical positive signal still does not clear the modern frozen gate |
| Resume gate | `frozen` | `resume_gate` | `FAIL: claim remains frozen` | `consistent` | frozen rows still replay to a frozen verdict |
| Forensic atlas | `frozen` | `forensic_atlas` | `FAIL: claim remains frozen` | `consistent` | frozen rows still replay to a frozen verdict |
| Final frozen state | `frozen` | `final_frozen_state` | `FAIL: claim remains frozen` | `consistent` | frozen rows still replay to a frozen verdict |
