# Forensic Route Locality Report

- phase-local counterfactual sample limit: `128`

## Phase-Local Route Statistics

| Seed | Group | Phase | Steps | Teacher Match | Route Entropy | Dominant Pair Frac | Unique Pair Count |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 47 | weak | at_key | 101 | 1.0000 | 1.3674 | 0.4230 | 4.7129 |
| 47 | weak | at_locked_door | 128 | 1.0000 | 1.3690 | 0.4381 | 4.5000 |
| 47 | weak | carry_key | 72 | 1.0000 | 1.3670 | 0.4083 | 4.7917 |
| 47 | weak | post_unlock | 15652 | 0.7663 | 1.3612 | 0.3828 | 4.2311 |
| 47 | weak | search_key | 47 | 1.0000 | 1.3673 | 0.4383 | 4.6383 |
| 53 | weak | at_key | 111 | 0.9640 | 1.3786 | 0.5198 | 4.0000 |
| 53 | weak | at_locked_door | 170 | 1.0000 | 1.3792 | 0.5336 | 4.0000 |
| 53 | weak | carry_key | 70 | 1.0000 | 1.3784 | 0.5423 | 4.0000 |
| 53 | weak | post_unlock | 6710 | 0.0382 | 1.3754 | 0.4613 | 4.0000 |
| 53 | weak | search_key | 48 | 1.0000 | 1.3787 | 0.5162 | 4.0000 |
| 59 | weak | at_key | 107 | 1.0000 | 1.3829 | 0.4368 | 4.4206 |
| 59 | weak | at_locked_door | 128 | 1.0000 | 1.3830 | 0.4438 | 4.1641 |
| 59 | weak | carry_key | 69 | 1.0000 | 1.3829 | 0.4388 | 4.3333 |
| 59 | weak | post_unlock | 5995 | 0.0427 | 1.3831 | 0.4801 | 4.0000 |
| 59 | weak | search_key | 45 | 1.0000 | 1.3826 | 0.4258 | 4.8667 |
| 7 | strong | at_key | 362 | 0.3094 | 1.3844 | 0.3592 | 5.2928 |
| 7 | strong | at_locked_door | 178 | 1.0000 | 1.3845 | 0.3565 | 5.3539 |
| 7 | strong | carry_key | 86 | 1.0000 | 1.3846 | 0.3600 | 5.7674 |
| 7 | strong | post_unlock | 283 | 1.0000 | 1.3844 | 0.3001 | 5.7774 |
| 7 | strong | search_key | 40 | 1.0000 | 1.3845 | 0.3600 | 5.7500 |
| 23 | strong | at_key | 109 | 1.0000 | 1.3807 | 0.3349 | 4.0000 |
| 23 | strong | at_locked_door | 128 | 1.0000 | 1.3809 | 0.3441 | 4.0000 |
| 23 | strong | carry_key | 59 | 1.0000 | 1.3807 | 0.3234 | 4.0000 |
| 23 | strong | post_unlock | 286 | 1.0000 | 1.3799 | 0.3355 | 4.0000 |
| 23 | strong | search_key | 43 | 1.0000 | 1.3806 | 0.3465 | 4.0000 |

## Weak-Seed SARE vs Single-Expert Phase Gaps

| Seed | Phase | SARE Teacher Match | single_expert Teacher Match | Gap |
| --- | --- | ---: | ---: | ---: |
| 47 | at_key | 1.0000 | 1.0000 | 0.0000 |
| 47 | at_locked_door | 1.0000 | 1.0000 | 0.0000 |
| 47 | carry_key | 1.0000 | 1.0000 | 0.0000 |
| 47 | post_unlock | 0.7663 | 0.5165 | -0.2498 |
| 47 | search_key | 1.0000 | 1.0000 | 0.0000 |
| 53 | at_key | 0.9640 | 0.9640 | 0.0000 |
| 53 | at_locked_door | 1.0000 | 1.0000 | 0.0000 |
| 53 | carry_key | 1.0000 | 1.0000 | 0.0000 |
| 53 | post_unlock | 0.0382 | 0.0382 | 0.0000 |
| 53 | search_key | 1.0000 | 1.0000 | 0.0000 |
| 59 | at_key | 1.0000 | 1.0000 | 0.0000 |
| 59 | at_locked_door | 1.0000 | 1.0000 | 0.0000 |
| 59 | carry_key | 1.0000 | 1.0000 | 0.0000 |
| 59 | post_unlock | 0.0427 | 0.0427 | 0.0000 |
| 59 | search_key | 1.0000 | 1.0000 | 0.0000 |

## Phase-Local Counterfactual Sensitivity

| Seed | Phase | Sample Count | Worst Ablation Change | Fixed-Router Change | All-Ablations Preserve |
| --- | --- | ---: | ---: | ---: | ---: |
| 47 | search_key | 47 | 0.8511 | 0.6170 | 0.0000 |
| 47 | at_key | 101 | 0.8416 | 0.8020 | 0.0000 |
| 47 | carry_key | 72 | 1.0000 | 0.9306 | 0.0000 |
| 47 | at_locked_door | 128 | 0.8359 | 0.6641 | 0.0000 |
| 47 | post_unlock | 128 | 0.5703 | 0.5703 | 0.0000 |
| 53 | at_key | 111 | 0.7928 | 0.6216 | 0.0000 |
| 53 | at_locked_door | 128 | 1.0000 | 0.3750 | 0.0000 |
| 53 | post_unlock | 128 | 1.0000 | 0.9688 | 0.0000 |
| 53 | search_key | 48 | 1.0000 | 0.6250 | 0.0000 |
| 53 | carry_key | 70 | 1.0000 | 0.6000 | 0.0000 |
| 59 | at_key | 107 | 0.8598 | 0.1776 | 0.1308 |
| 59 | at_locked_door | 128 | 0.8359 | 1.0000 | 0.0000 |
| 59 | post_unlock | 128 | 0.8438 | 0.2266 | 0.1562 |
| 59 | carry_key | 69 | 0.4058 | 1.0000 | 0.5942 |
| 59 | search_key | 45 | 0.2000 | 0.9111 | 0.7556 |
| 7 | search_key | 40 | 0.8500 | 0.0500 | 0.1500 |
| 7 | at_key | 128 | 0.7969 | 0.9375 | 0.2031 |
| 7 | at_locked_door | 128 | 0.8125 | 0.8125 | 0.1875 |
| 7 | post_unlock | 128 | 0.9297 | 1.0000 | 0.0000 |
| 7 | carry_key | 86 | 1.0000 | 1.0000 | 0.0000 |
| 23 | at_key | 109 | 0.6972 | 0.0367 | 0.2844 |
| 23 | at_locked_door | 128 | 0.8281 | 0.8281 | 0.1719 |
| 23 | post_unlock | 128 | 0.7188 | 0.2812 | 0.1328 |
| 23 | search_key | 43 | 0.6279 | 0.4884 | 0.2558 |
| 23 | carry_key | 59 | 1.0000 | 1.0000 | 0.0000 |

## Interpretation

- The weak block is not simply a low-route-usage story. Across `47/53/59`, routing stays concentrated and causally sensitive in phase-local slices, with mean dominant-pair fraction `0.4593` versus `0.3420` on the stronger comparison seeds.
- No weak-seed phase shows a clean local teacher-match win for matched `single_expert` over `SARE`. The sharper split comes from the trajectory casebook: seed `47` is the clearest route-fragile `SARE` failure, while seeds `53` and `59` mostly share the same late post-unlock collapse as matched `single_expert`.
- Seeds `53` and `59` look different. Their phase-local teacher-match rates are much closer to matched `single_expert`, which makes them look more like shared extraction failures than uniquely routed collapses.
- The stronger recovered seeds keep routing causally relevant in the same local phases, but they do so without the weak seeds' high dominant-pair concentration and long low-match post-unlock slices. That split supports a mixed mechanism story rather than one clean retry lever.
