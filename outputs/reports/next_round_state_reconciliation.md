# Next-Round State Reconciliation

- git commit: `251c92bc33d7b9ec5aa313dcbdd1a9c245bfa7fe`
- git dirty: `True`
- archived frozen pack: `outputs/reports/frozen_benchmark_pack.json`
- repaired operational gate reference pack: `outputs/reports/round6_current_benchmark_pack.json`
- live active benchmark pack: `outputs/reports/portfolio_candidate_pack.json`

## Reconciled Current Truth

- `round6` remains the active winning DoorKey line.
- The archived frozen pack remains the legacy provenance anchor and is no longer treated as the mutable live-state validator.
- The broad 80-run next-mega portfolio result narrowed the internal benchmark/frontier state instead of strengthening it.
- The public claim remains narrow: teacher-guided only, KL learner-state only, DoorKey only, external 64-episode evaluation only.

## Why Repair Was Needed

- The latest completed campaign ended with gate verdict `INCONCLUSIVE: missing prerequisites` because the archived frozen pack still sealed the live mutable `claim_ledger.md` path.
- That made later accepted-state ledger updates look like provenance drift even when the benchmark result itself had not changed.

## Repaired Operational Interpretation

- active benchmark: `round6`
- default restart prior: `round7`
- replay-validated alternate: `round10`
- hold-only prior: `['round5']`
- retired priors: `['door3_post5', 'post_unlock_x5']`
- live pack candidate: `round6`
- live pack gate reference: `outputs/reports/round6_current_benchmark_pack.json`

## Acceptance

- The repo now has one coherent pre-challenger state: archived frozen baseline for provenance, repaired current gate reference for live validation, and a narrowed `round6` operational benchmark state.
