# Control-First Audit Report

## What Was Wrong Or Uncertain

- `SyncVectorEnv` was using `NEXT_STEP` autoreset semantics, which meant the rollout loop could act on terminal observations and then store a garbage post-terminal transition on the next step.
- Time-limit truncations were being treated like true terminals for GAE/value bootstrapping, so truncated episodes lost their bootstrap target.
- Vector-env reset seeds overlapped across DDP ranks because the code passed scalar seeds back into Gymnasium reset instead of explicit per-env seed lists.
- Episode success was inferred from cumulative return rather than a durable per-episode success signal.
- Evaluation was only reported in greedy mode, which hid a large train/eval mismatch on some tasks.
- DDP episode metrics were reduced as unweighted per-rank means instead of sum/count aggregates.
- Resolved configs did not reflect CLI update overrides, and resume-from-checkpoint behavior was missing.
- In this environment, loopback sockets are not permitted, so the DDP smoke test must skip instead of pretending the harness regressed.

## What Was Fixed

- Vector env creation now uses `SAME_STEP` autoreset, and truncation bootstrapping uses `final_obs`.
- Reset seeds are explicit and non-overlapping across ranks.
- Episode success is carried through env info via `EpisodeSuccessWrapper`, with reward-based fallback only when needed.
- Evaluation runs on rank 0, restores RNG state after completion, and can be checked in greedy or sampled mode.
- Episode return/success/length logging now uses globally reduced sums and counts.
- Checkpoints now save update/step/RNG state, and training can resume from `--resume-from`.
- The DDP smoke test now skips when loopback sockets are unavailable.
- Canonical recovery configs now live under `configs/diagnostic/` and `configs/experiments/`; `configs/baseline/` remains the short-run smoke lane.

## What Baselines Learn Now

- Tiny overfit protocol on `MiniGrid-Empty-5x5-v0`:
  - `flat_dense`: greedy eval success `1.000`, eval return `0.955`
  - `token_dense`: greedy eval success `1.000`, eval return `0.955`
- Sanity sweep:
  - `Empty-5x5`: both `flat_dense` and `token_dense` solve it with greedy eval success `1.000`
  - `FourRooms`: `flat_dense` reaches weak but nonzero greedy eval (`0.0614` return, `0.0625` success); `token_dense` remains `0.000`
- Fully observed DoorKey diagnostic:
  - default control stayed flat at greedy eval `0.000`
  - tuned `token_dense` reached greedy eval return `0.964`, success `1.000`
- Standard partial-observation DoorKey:
  - tuned `token_dense` reached train success `1.000`
  - greedy eval stayed at `0.000`
  - sampled eval on the same checkpoint reached return `0.953`, success `1.000`
- Memory at 60 updates:
  - `token_gru` greedy eval stayed `0.000`, but sampled eval reached return `0.474`, success `0.500`
  - matched `token_dense` greedy eval stayed `0.000`, but sampled eval reached return `0.514`, success `0.5625`
  - caveat: `token_gru` is still diagnostic-only because PPO sequence-aware minibatching is deferred

## Whether Routed Models Were Fairly Tested

- The original routed sweeps were not fair tests of routing; the controls were still unhealthy.
- No new routed rerun is being claimed here.
- The current evidence says the repo was primarily an RL-baseline-debugging project until the control fixes above landed.

## Go / No-Go

- No-go on spending more time on TREG-H, SRW, or POR right now.
- No-go on claiming routed gains from the earlier MiniGrid suite.
- Conditional go on rerunning only `SARE`, and only after:
  - controls are matched to the tuned PPO setting,
  - evaluation mode is declared explicitly (`greedy` vs `sampled`),
  - compute is matched to the control.
- Highest-value next step is a compact PPO sweep and explicit greedy-vs-sampled policy diagnostics on DoorKey/KeyCorridor, not more architecture count.
