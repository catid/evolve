# Teacher-Extraction Decision Memo

## Decision

Pause routed greedy-performance work on DoorKey.

## Answers

### 1. Can `SARE` represent a competent greedy DoorKey policy under teacher-guided extraction?

Yes, conditionally.

Evidence:

- [learner_state_supervision_report.md](learner_state_supervision_report.md)
- [teacher_extraction_multiseed_report.md](teacher_extraction_multiseed_report.md)

The strongest learner-state supervised `SARE` runs reached:

- seed `7`: greedy success `0.5000`
- seed `11`: greedy success `1.0000`

So the current routed student is not blocked by an absolute capacity failure.

### 2. Is the blocker PPO/extraction path or routed-student limitation?

Both matter, but the evidence shifts the bottleneck away from pure student incapacity.

Evidence:

- [policy_distillation_report.md](policy_distillation_report.md)
- [learner_state_supervision_report.md](learner_state_supervision_report.md)
- [teacher_extraction_multiseed_report.md](teacher_extraction_multiseed_report.md)

Interpretation:

- PPO alone does not recover greedy `SARE`
- offline teacher distillation also does not recover greedy `SARE`
- learner-state supervision can recover greedy `SARE` on some seeds
- that same learner-state method fails completely on at least one seed

So the current state is:

- not a pure capacity no-go
- not a solved training-path problem either
- instead, a brittle extraction result that is not stable enough yet

### 3. Does the distillation pipeline work for tokenized students, fail for `SARE`, or fail generally?

Offline distillation fails generally.

Evidence:

- [policy_distillation_report.md](policy_distillation_report.md)

`flat_dense -> token_dense` and `flat_dense -> SARE` both stayed at greedy success `0.0000` under offline distillation, even when sampled behavior improved substantially.

Learner-state supervision is different:

- it did not rescue the tested `token_dense` student on the original seed-7 lane
- it did rescue `SARE` on seeds `7` and `11`

So the repo now has:

- a negative offline teacher-distillation result in general
- a partial positive learner-state result that is specific enough to `SARE` to be interesting, but not robust enough to treat as settled

### 4. Should routed work in this repo continue, pause, or stop?

Pause routed greedy-performance claims.

Evidence:

- [teacher_extraction_multiseed_report.md](teacher_extraction_multiseed_report.md)
- [distilled_route_integrity_best_seed_report.md](distilled_route_integrity_best_seed_report.md)

Reasoning:

- the best learner-state supervised `SARE` still uses routing in a meaningful way
- but the gain does not pass the repo’s small multi-seed robustness bar
- one routed seed fully succeeds, one partially succeeds, one fully fails

That is enough to say:

- a routed student can sometimes be extracted into a good greedy policy
- the current extraction method is too brittle to support a general routed claim

## Recommended Next Step

Only continue if the project is explicitly reframed as a bounded extraction-method effort with multi-seed validation as a hard gate. Otherwise, stop here and keep the current result as a partial, non-robust positive signal rather than a reopened routed win.
