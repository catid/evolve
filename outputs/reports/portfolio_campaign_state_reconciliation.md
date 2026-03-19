# Portfolio Campaign State Reconciliation

- status: `reconciled`
- active benchmark pack before this program: `outputs/reports/expansion_mega_program_candidate_pack.json`
- archived frozen pack: `outputs/reports/frozen_benchmark_pack.json`
- active benchmark candidate before this program: `round6`
- git commit: `85e2aaadae7c741e625431dbf494372007eb001c`
- git dirty: `True`

## Authoritative Current State

- `round6` is the active DoorKey benchmark before the portfolio program starts.
- `outputs/reports/frozen_benchmark_pack.json` remains the archived legacy baseline and provenance anchor.
- The allowed public claim envelope remains teacher-guided only, KL learner-state only, DoorKey only, and external 64-episode evaluation only.

## Reconciliation Checks

- docs present round6 as active benchmark: `True`
- claim ledger records the active benchmark state: `True`
- active benchmark pack names round6: `True`
- active benchmark pack archives frozen legacy baseline: `True`
- migration memo records active benchmark status: `True`
- mega-league memo records confirmed active benchmark status: `True`
- expansion-program memo records confirmed active benchmark status: `True`
- active benchmark gate verdict stays PASS: `True`

## Reconciled Interpretation

- This program starts from one coherent accepted state: `round6` is the active DoorKey benchmark, the frozen pack remains archived, and both stay comparable through the pack/gate lane.
- The portfolio campaign therefore begins from confirmation-plus-challenger evaluation rather than from a pending migration state.
