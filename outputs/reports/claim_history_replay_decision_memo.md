# Claim-History Replay Decision Memo

## Replay Outcome

- replay cases: `13`
- `PASS` verdicts on historical cases: `0`
- `FAIL` verdicts on historical cases: `6`
- `INCONCLUSIVE` verdicts on historical cases: `7`

## Gate vs History

- The current gate blocks the PPO-only negative family and the later over-broad multi-expert wording instead of letting them drift into thaw consideration.
- Early positive learner-state phases replay as missing-prerequisite or narrower cases rather than modern thaw candidates because they lack the final fairness controls and retry-block slice.
- The final frozen-era phases replay to the same frozen verdict the repo currently accepts.

## Ledger Alignment

- consistent ledger rows: `5`
- consistent-but-narrower rows: `6`
- inconsistent rows: `0`

## Final Result

- replay verdict map: `PASS: historical replay matches the expected claim-gate verdict map`
- snapshot status: `PASS`
- operational state: frozen DoorKey claim remains sealed, and the gate is now checked against both adversarial packs and the real historical claim trajectory.
