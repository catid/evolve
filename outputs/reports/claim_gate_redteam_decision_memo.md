# Claim Gate Red-Team Decision Memo

## Coverage

- adversarial corpus cases: `21`
- categories covered: happy-path reference, synthetic pass reference, incomplete packs, schema-invalid packs, semantically invalid packs, near-miss packs, provenance tampering packs, and overclaim packs

## Findings

- The gate now distinguishes `PASS`, `FAIL`, and `INCONCLUSIVE` across the adversarial corpus without relying on manual report inspection.
- Disallowed claim widening is now hard-failed even when paired with missing controls or malformed fields.
- Candidate pack tampering against `candidate_metrics_json` is now blocked by explicit consistency checks.
- Malformed structured inputs now yield concrete `INCONCLUSIVE` reasons instead of uncaught loader errors.

## Snapshot and Workflow

- golden snapshot status: `PASS`
- CI now runs the conformance suite in addition to the frozen-pack validation and full tests.

## Final Result

- conformance suite verdict: `PASS: claim-gate conformance suite matched the expected corpus verdicts`
- operational state: frozen DoorKey claim remains sealed behind the hardened pack-based gate
- thaw triage surface: candidate result pack + pack-based claim gate only
