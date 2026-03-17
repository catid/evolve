# Historical Candidate Pack: learner_state_supervision_initial

- family: `Learner-state supervision`
- ledger family: `Learner-state supervision`
- historical status: `bounded positive`
- expected verdict under current gate: `INCONCLUSIVE: missing prerequisites`
- rationale: The initial positive learner-state signal does not include the required matched controls, retry block, or combined DoorKey slice under the modern frozen gate.
- evaluation: `learner_state_round_summary` / `64` episodes
- requested claims: `['bounded_teacher_guided_doorkey_sare']`
- controls present: `['kl_lss_sare']`

## Source Reports

- `outputs/reports/learner_state_supervision_report.md`
- `outputs/reports/learner_state_supervision_report.csv`

## Provenance Note

This pack captures the first bounded positive learner-state signal, not a modern thaw-ready comparison pack.
