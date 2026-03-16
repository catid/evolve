# Greedy Recovery Decision Memo

## Question

Can architecture-neutral sharpening methods turn the existing sampled-competent `SARE` and `single_expert` DoorKey policies into competent greedy policies, without changing their routed structure?

## Starting Point

- reproduced baseline: [greedy_recovery_reproduction_note.md](greedy_recovery_reproduction_note.md)
- current control decomposition: [policy_extraction_report.md](policy_extraction_report.md)

Starting state:

- `flat_dense` greedy success `1.000`
- recovered `token_dense` greedy success `0.750`
- `single_expert` greedy success `0.000`, best sampled success `0.750`
- `SARE` greedy success `0.000`, best sampled success `1.000`

## Bounded Recovery Families Tested

1. Checkpoint selection:
   - [checkpoint_dynamics_report.md](checkpoint_dynamics_report.md)
2. Entropy schedules:
   - [entropy_schedule_report.md](entropy_schedule_report.md)
3. Self-imitation from successful sampled trajectories:
   - [self_imitation_report.md](self_imitation_report.md)

## Result

All three bounded recovery families were negative for greedy `SARE`.

- checkpoint selection: no nonzero greedy `SARE` checkpoint appeared anywhere in the archived DoorKey series
- entropy schedules: no tested schedule produced nonzero greedy `SARE`
- self-imitation: no tested fine-tune target or weighting produced nonzero greedy `SARE`

The same is true for `single_expert`.

## Decision

The current repo has reached a clean no-go result for this DoorKey greedy-recovery question.

- sampled competence alone is not converting into a usable greedy routed policy under the bounded architecture-neutral methods tested here
- current routed DoorKey claims should not be advanced as “almost working under greedy extraction”
- further routed work in this repo should pause unless a new extraction method is proposed and justified as a separate project

## Scope Not Taken

These were intentionally not run after the no-go outcome:

- optional margin-regularization probe
- KeyCorridor transfer check

That scope would have expanded the campaign after the core DoorKey question already had a clean answer.
