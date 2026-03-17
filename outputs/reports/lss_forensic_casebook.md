# Forensic Trajectory Casebook

- traced episodes per seed/variant: `64`
- trace path: deterministic single-env diagnostic rollouts seeded from the same base task seed; the external 64-episode policy-diagnostics path remains the final decision lane.

## Episode Summary

| Seed | Group | Variant | Success Rate | Failure Bucket | First Divergence Phase | Median First Divergence Step | Median Key Pickup Step | Median Unlock Step |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| 47 | weak | KL learner-state token_dense | 0.9688 | before_key_pickup | search_key | 0.0000 | 1.0000 | 4.0000 |
| 47 | weak | KL learner-state single_expert | 0.5469 | after_unlock_before_goal | post_unlock | 11.0000 | 1.0000 | 4.0000 |
| 47 | weak | KL learner-state SARE | 0.0000 | after_unlock_before_goal | post_unlock | 8.0000 | 1.0000 | 4.0000 |
| 53 | weak | KL learner-state token_dense | 1.0000 | success | search_key | 0.0000 | 1.0000 | 5.0000 |
| 53 | weak | KL learner-state single_expert | 0.5781 | after_unlock_before_goal | post_unlock | 10.0000 | 1.0000 | 5.0000 |
| 53 | weak | KL learner-state SARE | 0.5781 | after_unlock_before_goal | post_unlock | 10.0000 | 1.0000 | 5.0000 |
| 59 | weak | KL learner-state token_dense | 0.9531 | before_key_pickup | search_key | 1.0000 | 1.0000 | 4.0000 |
| 59 | weak | KL learner-state single_expert | 0.6250 | after_unlock_before_goal | post_unlock | 11.0000 | 1.0000 | 4.0000 |
| 59 | weak | KL learner-state SARE | 0.6250 | after_unlock_before_goal | post_unlock | 11.0000 | 1.0000 | 4.0000 |
| 7 | strong | KL learner-state token_dense | 1.0000 | success | - | - | 1.0000 | 6.0000 |
| 7 | strong | KL learner-state single_expert | 1.0000 | success | - | - | 1.0000 | 6.0000 |
| 7 | strong | KL learner-state SARE | 0.9844 | before_key_pickup | at_key | 0.0000 | 1.0000 | 6.0000 |
| 23 | strong | KL learner-state token_dense | 0.0000 | after_unlock_before_goal | post_unlock | 6.0000 | 1.0000 | 4.0000 |
| 23 | strong | KL learner-state single_expert | 0.3750 | after_unlock_before_goal | post_unlock | 9.0000 | 1.0000 | 4.0000 |
| 23 | strong | KL learner-state SARE | 1.0000 | success | - | - | 1.0000 | 4.0000 |

## Representative Aligned Episodes

### Seed `47` (`weak`; representative `sare_specific_failure` episode `3`)

| Variant | Success | Length | Failure Bucket | First Divergence Phase | Key Pickup Step | Unlock Step |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| KL learner-state token_dense | 1.0000 | 10.0000 | success | None | 2 | 5 |
| KL learner-state single_expert | 1.0000 | 10.0000 | success | None | 2 | 5 |
| KL learner-state SARE | 0.0000 | 250.0000 | after_unlock_before_goal | post_unlock | 2 | 5 |

#### KL learner-state token_dense

| Step | Phase | Teacher | Student | Match | Route Pair | Route Pair Frac |
| --- | --- | --- | --- | ---: | --- | ---: |
| 0 | at_key | `right` @ `0.984` | `right` @ `0.975` | 1.0000 | - | - |
| 1 | at_key | `right` @ `0.998` | `right` @ `0.956` | 1.0000 | - | - |
| 2 | at_key | `pickup` @ `0.981` | `pickup` @ `0.957` | 1.0000 | - | - |

#### KL learner-state single_expert

| Step | Phase | Teacher | Student | Match | Route Pair | Route Pair Frac |
| --- | --- | --- | --- | ---: | --- | ---: |
| 0 | at_key | `right` @ `0.984` | `right` @ `0.986` | 1.0000 | `[0]` | 1.0000 |
| 1 | at_key | `right` @ `0.998` | `right` @ `0.999` | 1.0000 | `[0]` | 1.0000 |
| 2 | at_key | `pickup` @ `0.981` | `pickup` @ `0.979` | 1.0000 | `[0]` | 1.0000 |

#### KL learner-state SARE

| Step | Phase | Teacher | Student | Match | Route Pair | Route Pair Frac |
| --- | --- | --- | --- | ---: | --- | ---: |
| 7 | post_unlock | `forward` @ `0.999` | `forward` @ `0.974` | 1.0000 | `[1, 3]` | 0.3600 |
| 8 | post_unlock | `right` @ `0.999` | `right` @ `0.999` | 1.0000 | `[1, 3]` | 0.4400 |
| 9 | post_unlock | `forward` @ `0.999` | `right` @ `0.470` | 0.0000 | `[1, 3]` | 0.3400 |
| 10 | post_unlock | `left` @ `0.911` | `forward` @ `0.952` | 0.0000 | `[1, 3]` | 0.3600 |
| 11 | post_unlock | `left` @ `0.577` | `forward` @ `0.987` | 0.0000 | `[1, 3]` | 0.3600 |

### Seed `53` (`weak`; representative `shared_structured_failure` episode `0`)

| Variant | Success | Length | Failure Bucket | First Divergence Phase | Key Pickup Step | Unlock Step |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| KL learner-state token_dense | 1.0000 | 10.0000 | success | None | 0 | 4 |
| KL learner-state single_expert | 0.0000 | 250.0000 | after_unlock_before_goal | post_unlock | 0 | 4 |
| KL learner-state SARE | 0.0000 | 250.0000 | after_unlock_before_goal | post_unlock | 0 | 4 |

#### KL learner-state token_dense

| Step | Phase | Teacher | Student | Match | Route Pair | Route Pair Frac |
| --- | --- | --- | --- | ---: | --- | ---: |
| 0 | at_key | `pickup` @ `0.999` | `pickup` @ `0.997` | 1.0000 | - | - |
| 1 | at_locked_door | `right` @ `0.841` | `right` @ `0.926` | 1.0000 | - | - |
| 2 | at_locked_door | `right` @ `0.994` | `right` @ `0.965` | 1.0000 | - | - |

#### KL learner-state single_expert

| Step | Phase | Teacher | Student | Match | Route Pair | Route Pair Frac |
| --- | --- | --- | --- | ---: | --- | ---: |
| 7 | post_unlock | `right` @ `1.000` | `right` @ `0.953` | 1.0000 | `[0]` | 1.0000 |
| 8 | post_unlock | `forward` @ `1.000` | `forward` @ `0.999` | 1.0000 | `[0]` | 1.0000 |
| 9 | post_unlock | `forward` @ `0.998` | `right` @ `0.811` | 0.0000 | `[0]` | 1.0000 |
| 10 | post_unlock | `left` @ `0.974` | `forward` @ `0.964` | 0.0000 | `[0]` | 1.0000 |
| 11 | post_unlock | `left` @ `0.974` | `forward` @ `0.964` | 0.0000 | `[0]` | 1.0000 |

#### KL learner-state SARE

| Step | Phase | Teacher | Student | Match | Route Pair | Route Pair Frac |
| --- | --- | --- | --- | ---: | --- | ---: |
| 7 | post_unlock | `right` @ `1.000` | `right` @ `0.993` | 1.0000 | `[0, 3]` | 0.5000 |
| 8 | post_unlock | `forward` @ `1.000` | `forward` @ `1.000` | 1.0000 | `[0, 3]` | 0.4600 |
| 9 | post_unlock | `forward` @ `0.998` | `right` @ `0.942` | 0.0000 | `[0, 3]` | 0.5200 |
| 10 | post_unlock | `left` @ `0.974` | `forward` @ `0.999` | 0.0000 | `[0, 3]` | 0.4600 |
| 11 | post_unlock | `left` @ `0.974` | `forward` @ `0.999` | 0.0000 | `[0, 3]` | 0.4600 |

### Seed `59` (`weak`; representative `shared_structured_failure` episode `1`)

| Variant | Success | Length | Failure Bucket | First Divergence Phase | Key Pickup Step | Unlock Step |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| KL learner-state token_dense | 1.0000 | 10.0000 | success | None | 1 | 4 |
| KL learner-state single_expert | 0.0000 | 250.0000 | after_unlock_before_goal | post_unlock | 1 | 4 |
| KL learner-state SARE | 0.0000 | 250.0000 | after_unlock_before_goal | post_unlock | 1 | 4 |

#### KL learner-state token_dense

| Step | Phase | Teacher | Student | Match | Route Pair | Route Pair Frac |
| --- | --- | --- | --- | ---: | --- | ---: |
| 0 | at_key | `right` @ `0.985` | `right` @ `0.982` | 1.0000 | - | - |
| 1 | at_key | `pickup` @ `0.995` | `pickup` @ `0.995` | 1.0000 | - | - |
| 2 | carry_key | `forward` @ `0.998` | `forward` @ `0.998` | 1.0000 | - | - |

#### KL learner-state single_expert

| Step | Phase | Teacher | Student | Match | Route Pair | Route Pair Frac |
| --- | --- | --- | --- | ---: | --- | ---: |
| 7 | post_unlock | `right` @ `0.999` | `right` @ `0.999` | 1.0000 | `[0]` | 1.0000 |
| 8 | post_unlock | `forward` @ `1.000` | `forward` @ `0.996` | 1.0000 | `[0]` | 1.0000 |
| 9 | post_unlock | `forward` @ `0.994` | `pickup` @ `0.957` | 0.0000 | `[0]` | 1.0000 |
| 10 | post_unlock | `forward` @ `0.994` | `pickup` @ `0.957` | 0.0000 | `[0]` | 1.0000 |
| 11 | post_unlock | `forward` @ `0.994` | `pickup` @ `0.957` | 0.0000 | `[0]` | 1.0000 |

#### KL learner-state SARE

| Step | Phase | Teacher | Student | Match | Route Pair | Route Pair Frac |
| --- | --- | --- | --- | ---: | --- | ---: |
| 7 | post_unlock | `right` @ `0.999` | `right` @ `0.999` | 1.0000 | `[0, 1]` | 0.5000 |
| 8 | post_unlock | `forward` @ `1.000` | `forward` @ `1.000` | 1.0000 | `[0, 1]` | 0.5200 |
| 9 | post_unlock | `forward` @ `0.994` | `pickup` @ `0.763` | 0.0000 | `[0, 1]` | 0.4800 |
| 10 | post_unlock | `forward` @ `0.994` | `pickup` @ `0.763` | 0.0000 | `[0, 1]` | 0.4800 |
| 11 | post_unlock | `forward` @ `0.994` | `pickup` @ `0.763` | 0.0000 | `[0, 1]` | 0.4800 |

### Seed `7` (`strong`; representative `strong_sare_success` episode `0`)

| Variant | Success | Length | Failure Bucket | First Divergence Phase | Key Pickup Step | Unlock Step |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| KL learner-state token_dense | 1.0000 | 11.0000 | success | None | 2 | 6 |
| KL learner-state single_expert | 1.0000 | 11.0000 | success | None | 2 | 6 |
| KL learner-state SARE | 1.0000 | 11.0000 | success | None | 2 | 6 |

#### KL learner-state token_dense

| Step | Phase | Teacher | Student | Match | Route Pair | Route Pair Frac |
| --- | --- | --- | --- | ---: | --- | ---: |
| 0 | search_key | `right` @ `0.998` | `right` @ `0.999` | 1.0000 | - | - |
| 1 | search_key | `forward` @ `0.995` | `forward` @ `0.995` | 1.0000 | - | - |
| 2 | at_key | `pickup` @ `0.988` | `pickup` @ `0.988` | 1.0000 | - | - |

#### KL learner-state single_expert

| Step | Phase | Teacher | Student | Match | Route Pair | Route Pair Frac |
| --- | --- | --- | --- | ---: | --- | ---: |
| 0 | search_key | `right` @ `0.998` | `right` @ `0.998` | 1.0000 | `[0]` | 1.0000 |
| 1 | search_key | `forward` @ `0.995` | `forward` @ `0.992` | 1.0000 | `[0]` | 1.0000 |
| 2 | at_key | `pickup` @ `0.988` | `pickup` @ `0.995` | 1.0000 | `[0]` | 1.0000 |

#### KL learner-state SARE

| Step | Phase | Teacher | Student | Match | Route Pair | Route Pair Frac |
| --- | --- | --- | --- | ---: | --- | ---: |
| 0 | search_key | `right` @ `0.998` | `right` @ `0.999` | 1.0000 | `[0, 3]` | 0.3600 |
| 1 | search_key | `forward` @ `0.995` | `forward` @ `0.994` | 1.0000 | `[2, 3]` | 0.3600 |
| 2 | at_key | `pickup` @ `0.988` | `pickup` @ `0.986` | 1.0000 | `[2, 3]` | 0.3600 |

### Seed `23` (`strong`; representative `strong_sare_success` episode `0`)

| Variant | Success | Length | Failure Bucket | First Divergence Phase | Key Pickup Step | Unlock Step |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| KL learner-state token_dense | 0.0000 | 250.0000 | after_unlock_before_goal | post_unlock | 1 | 3 |
| KL learner-state single_expert | 1.0000 | 8.0000 | success | None | 1 | 3 |
| KL learner-state SARE | 1.0000 | 8.0000 | success | None | 1 | 3 |

#### KL learner-state token_dense

| Step | Phase | Teacher | Student | Match | Route Pair | Route Pair Frac |
| --- | --- | --- | --- | ---: | --- | ---: |
| 3 | at_locked_door | `toggle` @ `0.996` | `toggle` @ `0.995` | 1.0000 | - | - |
| 4 | post_unlock | `forward` @ `0.999` | `forward` @ `0.996` | 1.0000 | - | - |
| 5 | post_unlock | `forward` @ `0.999` | `left` @ `0.948` | 0.0000 | - | - |
| 6 | post_unlock | `forward` @ `0.772` | `right` @ `0.939` | 0.0000 | - | - |
| 7 | post_unlock | `forward` @ `0.999` | `left` @ `0.948` | 0.0000 | - | - |

#### KL learner-state single_expert

| Step | Phase | Teacher | Student | Match | Route Pair | Route Pair Frac |
| --- | --- | --- | --- | ---: | --- | ---: |
| 0 | at_key | `right` @ `0.982` | `right` @ `0.973` | 1.0000 | `[0]` | 1.0000 |
| 1 | at_key | `pickup` @ `0.995` | `pickup` @ `0.996` | 1.0000 | `[0]` | 1.0000 |
| 2 | at_locked_door | `left` @ `0.998` | `left` @ `0.999` | 1.0000 | `[0]` | 1.0000 |

#### KL learner-state SARE

| Step | Phase | Teacher | Student | Match | Route Pair | Route Pair Frac |
| --- | --- | --- | --- | ---: | --- | ---: |
| 0 | at_key | `right` @ `0.982` | `right` @ `0.975` | 1.0000 | `[1, 3]` | 0.3200 |
| 1 | at_key | `pickup` @ `0.995` | `pickup` @ `0.997` | 1.0000 | `[0, 3]` | 0.3400 |
| 2 | at_locked_door | `left` @ `0.998` | `left` @ `1.000` | 1.0000 | `[0, 3]` | 0.3400 |

## Interpretation

- Weak-block `SARE` failures are not uniform. On this traced slice, SARE-specific underperformance versus matched `single_expert` is concentrated on seeds `[47]`, while seeds `[53, 59]` look more like shared structured-student failures against a stronger tokenized control.
- The stronger comparison seeds keep recovered `SARE` fully successful on the traced slice (`7:0.984, 23:1.000`), and they reach key-pickup / unlock milestones without the long post-unlock loops seen in the weak block.
- The representative aligned episodes show where the aggregate audit was too coarse: the weak block does not fail in one common place. Some episodes are clearly route-fragile, while others fail in the same late post-unlock phase that also troubles matched single_expert. That split argues against one clean retry mechanism across the whole `47/53/59` block.
