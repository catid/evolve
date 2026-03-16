# Learner-State Robustness Decision Memo

## Decision Path

- final metric path: external `policy_diagnostics`
- evaluation budget: `64` episodes per mode
- comparison seeds: `7`, `11`, `19`

## What Was Reproduced

- Baseline reproduction is in [lss_robustness_reproduction_note.md](lss_robustness_reproduction_note.md).
- The reproduced external-greedy baseline remains:
  - `flat_dense`: `1.0000`, `1.0000`, `1.0000`
  - recovered `token_dense`: `0.7031`, `0.0000`, `1.0000`
  - baseline PPO `SARE`: `0.0000`, `0.0000`, `0.0000`
  - hard-label learner-state `SARE`: `0.5000`, `1.0000`, `0.0000`

## What The Diagnostics Show

- Seed heterogeneity is summarized in [lss_seed_heterogeneity_report.md](lss_seed_heterogeneity_report.md).
- The failed hard-label seed is not mainly explained by weak teacher confidence. By later rounds, teacher confidence is high on all seeds.
- The more plausible failure mode is optimization around the learner-state dataset: the winning seed is the only one whose final round sharply reduces the collected dataset size and increases state-diversity ratio, while failed seeds keep training on the full append-all pile.

## What The Robustness Sweep Showed

- Sweep results are in [lss_robustness_sweep_report.md](lss_robustness_sweep_report.md).
- The only improving method family was teacher-logit KL supervision with the original `append_all` aggregation.
- `cap_recent` and `cap_recent_balanced` were both negative.
- The minimal tokenized-student sanity check on seed `19` was positive, so the stronger learner-state method is not specific to routed students.

## Hard Gate Result

- The external 3-seed gate is in [lss_robustness_multiseed_report.md](lss_robustness_multiseed_report.md).
- Improved KL learner-state `SARE` reaches greedy success:
  - seed `7`: `1.0000`
  - seed `11`: `0.5625`
  - seed `19`: `0.5781`
- Mean greedy success is `0.7135`, which beats recovered `token_dense` mean greedy success `0.5677`.
- No seed remains at greedy success `0.0`.

## Route Integrity

- Route integrity on the revived seed is in [lss_route_integrity_report.md](lss_route_integrity_report.md).
- On seed `19`, the improved method keeps route entropy and active compute essentially unchanged while lifting greedy success to `0.5938`.
- Expert loads remain spread across all four experts, so the reopened claim is not an obvious route-collapse artifact.

## Final Answer

1. Is learner-state supervision for `SARE` robust enough to reopen routed greedy-performance claims?

Yes, for DoorKey under teacher-guided extraction. The KL learner-state method passes the repo’s 3-seed external gate.

2. What was the main source of seed brittleness?

The main issue was target choice plus learner-state optimization, not weak teacher confidence. Hard action labels were too brittle; teacher-logit KL supervision was the only bounded change that robustly lifted the previously failed seed.

3. Which improvement mattered most?

Teacher-logit KL supervision. Aggregation changes did not help and usually hurt.

4. Did the final method match or beat recovered `token_dense` on the 3-seed external gate?

Yes. `SARE` with KL learner-state supervision reached mean greedy success `0.7135`, above recovered `token_dense` at `0.5677`, and eliminated complete-seed failure.

5. Should routed work in this repo continue, pause, or stop?

Continue, but only under a narrow claim:
- this is a teacher-guided extraction result, not a PPO-alone routed win
- DoorKey is the only task on which the routed greedy claim is reopened
- future routed claims should keep the external multi-seed gate and route-integrity check as hard requirements
