# Claim Gate Dry Run

- manifest: `configs/claims/doorkey_frozen_claim.yaml`
- candidate: `outputs/reports/frozen_baseline_validation.json`
- current frozen status: `frozen`

| Check | Result | Detail |
| --- | --- | --- |
| task | `PASS` | candidate task matches `DoorKey` |
| evaluation_path | `PASS` | candidate uses `external_policy_diagnostics` with `64` episodes |
| fairness_controls | `PASS` | all required structured controls are present |
| claim_scope | `PASS` | candidate stays inside the frozen claim envelope |
| candidate_metrics | `PASS` | candidate exposes retry-block and combined metrics for required variants |
| retry_block_improvement | `FAIL` | candidate retry-block SARE mean `0.3125` does not exceed frozen baseline `0.3125` |
| retry_block_vs_single_expert | `FAIL` | candidate retry-block SARE mean `0.3125` trails same-block single_expert `0.4635` |
| retry_block_failures | `PASS` | candidate retry-block SARE complete-seed failures `1` stay within gate `1` |
| combined_picture_mean | `PASS` | candidate combined SARE mean `0.7122` preserves or improves frozen baseline `0.7122` |
| combined_picture_failures | `PASS` | candidate combined SARE complete-seed failures `1` stay within gate `1` |

## Verdict

FAIL: claim remains frozen
