# Resume-Gate Failure Mechanism Report

## Reference Comparison

| Group | Teacher Conf | Disagreement | Unique Ratio | Route Entropy | Path Entropy | Dominant Route Pair |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stronger recovered `SARE` seeds | `0.9582` | `0.5811` | `0.0038` | `1.3765` | `1.0736` | `0.3461` |
| final block `47/53/59` | `0.9828` | `0.7815` | `0.0026` | `1.3817` | `0.7722` | `see per-seed below` |

## Per-Seed Final-Block Summary

| Seed | Variant | Best Round | Best Greedy | Final Greedy | Disagreement r1->rN | Teacher Conf r1->rN | Student Conf r1->rN | Unique Ratio r1->rN | Route Ent r1->rN | Path Ent r1->rN | Dominant Route Pair |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | ---: |
| 47 | KL learner-state SARE | 1 | 0.0000 | 0.0000 | `1.0000->0.9747` | `0.9911->0.9962` | `0.3272->0.7746` | `0.0016->0.0033` | `1.3846->1.3706` | `0.5627->0.7764` | 0.7128 |
| 47 | KL learner-state single_expert | 4 | 0.4531 | 0.4531 | `1.0000->0.6307` | `0.9911->0.9535` | `0.8334->0.9065` | `0.0016->0.0035` | `0.0000->0.0000` | `0.0000->0.0000` | `n/a` |
| 53 | KL learner-state SARE | 3 | 0.5156 | 0.5156 | `0.9996->0.2359` | `0.9969->0.9609` | `0.2476->0.9182` | `0.0018->0.0066` | `1.3857->1.3786` | `0.4854->0.7414` | 0.5852 |
| 53 | KL learner-state single_expert | 3 | 0.5156 | 0.5156 | `0.9992->0.2359` | `0.9970->0.9609` | `0.3840->0.9529` | `0.0019->0.0066` | `0.0000->0.0000` | `0.0000->0.0000` | `n/a` |
| 59 | KL learner-state SARE | 4 | 0.4219 | 0.4219 | `0.6367->0.9715` | `0.9858->0.9992` | `0.5413->0.8045` | `0.0023->0.0032` | `1.3857->1.3833` | `0.5310->1.0304` | 0.4548 |
| 59 | KL learner-state single_expert | 4 | 0.4219 | 0.4219 | `0.9991->0.9752` | `0.9944->0.9969` | `0.3804->0.9020` | `0.0021->0.0032` | `0.0000->0.0000` | `0.0000->0.0000` | `n/a` |

## Mechanism Audit

- Teacher quality is not the blocker: best-round teacher confidence stays high on the final block (`0.9828` vs `0.9582` on stronger recovered seeds).
- Learner-state coverage is lower on the final block (`0.0026` vs `0.0038`), so coverage could contribute.
- State-local expert redundancy is elevated across most final-block seeds, with dominant route-pair concentration above the stronger-seed reference on multiple seeds.
- Route-specific fragility remains real, not absent: fixed-router override and worst-expert ablation still collapse the recovered `53` and `59` checkpoints to zero in the prior final-block probe set.
- The main fairness signal cuts against a resume attempt: matched KL learner-state `single_expert` is already at least as strong on the final block (`0.4635` vs `0.3125` for `SARE`).
- The weak block suggests one plausible mechanism, but it is not strong or consistent enough yet to justify a resume attempt.

## Verdict

mechanism plausible but weak
