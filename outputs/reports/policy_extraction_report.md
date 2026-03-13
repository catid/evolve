# Policy Extraction Report

## Question

Why does `flat_dense` solve greedy DoorKey while the original tokenized path does not?

Artifact root for this phase:

- [outputs/diagnostics/policy_extraction/doorkey/report.md](/home/catid/evolve/outputs/diagnostics/policy_extraction/doorkey/report.md)
- [outputs/reports/policy_extraction_report.csv](/home/catid/evolve/outputs/reports/policy_extraction_report.csv)

## Results

| Variant | Greedy Success | Best Sampled Success | Greedy Max Prob | Greedy Margin | Train Return |
| --- | ---: | ---: | ---: | ---: | ---: |
| `flat_dense` | `1.000` | `1.000` | `0.995` | `6.671` | `0.962` |
| `token_dense` | `0.000` | `0.125` | `0.315` | `0.330` | `0.310` |
| `single_expert` | `0.000` | `1.000` | `0.353` | `0.370` | `0.536` |
| `SARE` | `0.000` | `1.000` | `0.391` | `0.369` | `0.639` |

## Interpretation

- `flat_dense` learns a sharp policy. Its greedy and sampled evaluation are effectively identical.
- `token_dense` is not just a greedy-extraction casualty. It is weak even under sampled evaluation, which means the tokenized dense path itself is part of the bottleneck.
- `single_expert` and `SARE` are different: both learn policies that are strong under sampled evaluation but too soft for greedy extraction. Their problem is much more about action calibration than raw train-time solvability.

## Answer

The flat-dense vs tokenized gap is not one bug.

- `token_dense` is weaker in both representation and extraction.
- `single_expert` and `SARE` are closer to good policies, but their action distributions stay too broad for greedy evaluation.

That is why the next phase had to focus on tokenized-control recovery and not on adding more routed variants.
