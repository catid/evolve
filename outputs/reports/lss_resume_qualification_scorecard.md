# Resume Qualification Scorecard

| Candidate | Evidence For | Evidence Against | Bounded KL-LSS Intervention? | Confidence |
| --- | --- | --- | --- | --- |
| teacher-quality problem | Teacher confidence remains very high on the weak block, so there is little evidence that labels are noisy. | High teacher confidence on `47/53/59` is evidence against this mechanism, not for it. | no | low |
| learner-state coverage problem | Weak final rounds keep lower unique-state ratio (`0.0044` vs `0.0837`) and never reach the strong seeds' short successful cleanup rounds. | Coverage improves somewhat inside the weak block without delivering one shared recovery pattern across `47/53/59`. | possible but underspecified | medium |
| append-all accumulation / stale-data problem | Weak seeds keep appending max-length failed collections, while strong seeds transition into short, high-coverage late rounds. | The same append-all path does not prevent matched `single_expert` from beating `SARE` on seed `47`, so stale accumulation is not a sufficient explanation by itself. | possible but underspecified | medium |
| checkpoint-selection problem | Some weak seeds have different best-round and final-round disagreement profiles. | Checkpoint selection is already a closed negative family in this repo, and the round audit still does not reveal a missed clearly superior external-greedy checkpoint on `47/53/59`. | no | low |
| route-specific fragility problem | Phase-local analysis shows `SARE` losing teacher match to matched `single_expert` on weak seeds, and counterfactual routing perturbations still change actions materially (mean fixed-router action-change `0.6610`). | That fragility is not consistent across all weak seeds: seed `47` is clearly route-fragile, while `53/59` look closer to shared structured-student failures. | possible but seed-split | medium |
| state-local expert redundancy problem | Weak seeds finish with higher dominant route-pair concentration (`0.4686` vs `0.3422`), and matched `single_expert` stays close overall (`teacher-match gap -0.0167`). | The stronger recovered seeds remain causally routing-dependent in the same local phases, so redundancy does not cleanly explain the whole weak block. | possible but mixed | medium |
| none clearly isolated | The casebook, round audit, and route-locality pass all split the weak block: seed `47` is route-fragile, while `53/59` look closer to general extraction mismatch. | There are still real recurring signals around disagreement persistence, stale failed rounds, and local route concentration. | n/a | high |

## Verdict

bounded retry not justified
