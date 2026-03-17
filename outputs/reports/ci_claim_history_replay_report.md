# CI Claim History Replay Report

- workflow: `.github/workflows/claim-gate.yml`
- replay verdict: `PASS: historical replay matches the expected claim-gate verdict map`
- snapshot status: `PASS`
- historical cases covered: `13`

## Workflow Coverage

- validates the frozen benchmark pack
- runs the adversarial claim-gate conformance corpus
- replays the gate against the real historical candidate-pack catalog
- checks the historical replay golden snapshot for drift
- runs the full test suite
