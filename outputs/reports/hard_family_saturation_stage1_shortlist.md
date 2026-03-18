# Saturation Hard-Family Stage 1 Shortlist

| Candidate | Intervention Family | Mechanism Hypothesis | Hard-Family Target | Broader DoorKey Risk |
| --- | --- | --- | --- | --- |
| `post_unlock_x5` | post_unlock_weight_schedule | stronger post-unlock KL alone may finish the late cleanup states that still sink routed SARE on the hard family. | pushes more KL mass onto the exact post-unlock states where post_pass_b and post_pass_c still trail token_dense. | overweights late cleanup and can flatten healthy-block route diversity. |

## Interpretation

- The shortlist stays inside the existing teacher-guided KL learner-state family.
- The saturation program covers ten distinct mechanism directions with twenty bounded variants before narrowing, so a negative result would reflect a real family-level search inside the existing learner-state surface rather than another one-off tweak.
| `post_unlock_x6` | post_unlock_weight_schedule | a still stronger post-unlock KL may close the remaining hard-family token gap when x5 is not enough. | attacks the same late cleanup states with a more aggressive post-unlock emphasis. | can overfit late states and create brittle checkpoint spikes. |

## Interpretation

- The shortlist stays inside the existing teacher-guided KL learner-state family.
- The saturation program covers ten distinct mechanism directions with twenty bounded variants before narrowing, so a negative result would reflect a real family-level search inside the existing learner-state surface rather than another one-off tweak.
| `door2_post4` | locked_door_bridge_weighting | weighting the approach-to-door phase may preserve the context needed for post-unlock cleanup on hard-family seeds. | keeps at_locked_door states visible before the post-unlock correction begins. | may spend KL budget on already-solved door-approach states and blunt the late-phase gain. |

## Interpretation

- The shortlist stays inside the existing teacher-guided KL learner-state family.
- The saturation program covers ten distinct mechanism directions with twenty bounded variants before narrowing, so a negative result would reflect a real family-level search inside the existing learner-state surface rather than another one-off tweak.
| `door3_post5` | locked_door_bridge_weighting | a heavier locked-door bridge plus stronger post-unlock weight may improve late-phase continuity on the hard family. | biases the learner-state loop toward the transition from locked door to successful cleanup. | can overweight a narrow phase transition and miss healthy-block exploration context. |

## Interpretation

- The shortlist stays inside the existing teacher-guided KL learner-state family.
- The saturation program covers ten distinct mechanism directions with twenty bounded variants before narrowing, so a negative result would reflect a real family-level search inside the existing learner-state surface rather than another one-off tweak.
| `carry2_post4` | carry_key_bridge_weighting | keeping carry-key states slightly hotter may reduce hard-family late collapse by preserving pre-door context. | compensates for dev seeds where the routed student drifts before door interaction and pays for it after unlock. | may dilute the post-unlock fix and help healthy blocks more than the hard family. |

## Interpretation

- The shortlist stays inside the existing teacher-guided KL learner-state family.
- The saturation program covers ten distinct mechanism directions with twenty bounded variants before narrowing, so a negative result would reflect a real family-level search inside the existing learner-state surface rather than another one-off tweak.
| `carry3_post5` | carry_key_bridge_weighting | a stronger carry-key bridge plus stronger post-unlock weight may stabilize the full late trajectory on hard-family seeds. | emphasizes the final approach-to-door sequence before the problematic cleanup segment. | can turn into a generic structured-student improvement that does not help routed SARE specifically. |

## Interpretation

- The shortlist stays inside the existing teacher-guided KL learner-state family.
- The saturation program covers ten distinct mechanism directions with twenty bounded variants before narrowing, so a negative result would reflect a real family-level search inside the existing learner-state surface rather than another one-off tweak.
| `disagree050` | disagreement_threshold_weighting | a modest disagreement bonus can amplify the exact late states where routed SARE still diverges from the teacher on the hard family. | focuses learning on post-unlock disagreement pockets without changing the extraction family. | may overreact to noisy disagreement and destabilize healthy blocks. |

## Interpretation

- The shortlist stays inside the existing teacher-guided KL learner-state family.
- The saturation program covers ten distinct mechanism directions with twenty bounded variants before narrowing, so a negative result would reflect a real family-level search inside the existing learner-state surface rather than another one-off tweak.
| `disagree100` | disagreement_threshold_weighting | a larger disagreement bonus may be needed if the hard-family mismatch is still too sparse for weaker bonuses. | saturates the late disagreement signal that remains after the current post-unlock weighting. | can chase transient disagreement spikes instead of durable routed behavior. |

## Interpretation

- The shortlist stays inside the existing teacher-guided KL learner-state family.
- The saturation program covers ten distinct mechanism directions with twenty bounded variants before narrowing, so a negative result would reflect a real family-level search inside the existing learner-state surface rather than another one-off tweak.
| `teacher_conf_post4` | late_confidence_weighting | teacher-confidence weighting can prioritize the high-certainty late states that routed SARE still misses on hard-family seeds. | focuses KL on post-unlock states where the teacher is decisive and the student still lags. | may erase useful low-confidence correction states and help token_dense more than routed SARE. |

## Interpretation

- The shortlist stays inside the existing teacher-guided KL learner-state family.
- The saturation program covers ten distinct mechanism directions with twenty bounded variants before narrowing, so a negative result would reflect a real family-level search inside the existing learner-state surface rather than another one-off tweak.
| `teacher_conf_post5` | late_confidence_weighting | stronger post-unlock confidence weighting may be needed if the hard-family gap is concentrated in very decisive cleanup states. | couples teacher-confidence and stronger post-unlock emphasis on the same late states. | can collapse the learner-state dataset onto a tiny late slice and produce brittle wins. |

## Interpretation

- The shortlist stays inside the existing teacher-guided KL learner-state family.
- The saturation program covers ten distinct mechanism directions with twenty bounded variants before narrowing, so a negative result would reflect a real family-level search inside the existing learner-state surface rather than another one-off tweak.
| `round5` | cleanup_round_extension | one extra learner-state cleanup round may finish near-miss hard-family episodes without changing the base weighting shape. | extends the existing post-unlock candidate through one more correction round. | can create a narrow late-round checkpoint spike and regress healthy blocks. |

## Interpretation

- The shortlist stays inside the existing teacher-guided KL learner-state family.
- The saturation program covers ten distinct mechanism directions with twenty bounded variants before narrowing, so a negative result would reflect a real family-level search inside the existing learner-state surface rather than another one-off tweak.
| `round6` | cleanup_round_extension | two extra rounds may be needed if the hard-family fix requires a longer cleanup tail than the current candidate allows. | further extends late cleanup on the same learner-state path. | makes stale data accumulation worse if the family is not actually round-limited. |

## Interpretation

- The shortlist stays inside the existing teacher-guided KL learner-state family.
- The saturation program covers ten distinct mechanism directions with twenty bounded variants before narrowing, so a negative result would reflect a real family-level search inside the existing learner-state surface rather than another one-off tweak.
| `phase_balanced_4096` | phase_balanced_recent_replay | a bounded phase-balanced recent buffer can protect hard-family late states from stale append-all domination. | preserves post-unlock coverage while trimming long failed histories. | may discard useful early context on healthy blocks. |

## Interpretation

- The shortlist stays inside the existing teacher-guided KL learner-state family.
- The saturation program covers ten distinct mechanism directions with twenty bounded variants before narrowing, so a negative result would reflect a real family-level search inside the existing learner-state surface rather than another one-off tweak.
| `phase_balanced_6144` | phase_balanced_recent_replay | a slightly larger phase-balanced buffer may keep more useful context while still protecting late hard-family states. | softens stale-data pressure without shrinking the replay window too aggressively. | can drift back toward append-all behavior and lose the hard-family benefit. |

## Interpretation

- The shortlist stays inside the existing teacher-guided KL learner-state family.
- The saturation program covers ten distinct mechanism directions with twenty bounded variants before narrowing, so a negative result would reflect a real family-level search inside the existing learner-state surface rather than another one-off tweak.
| `cap_balanced_4096` | balanced_recent_replay | a balanced recent cap may control hard-family stale-data buildup without the stronger phase quotas. | trims old failed episodes while keeping the recent late-phase trajectory mix. | may be too weak to close the token gap and still hurt healthy-block context. |

## Interpretation

- The shortlist stays inside the existing teacher-guided KL learner-state family.
- The saturation program covers ten distinct mechanism directions with twenty bounded variants before narrowing, so a negative result would reflect a real family-level search inside the existing learner-state surface rather than another one-off tweak.
| `cap_balanced_6144` | balanced_recent_replay | a larger balanced cap may give the hard family enough late coverage without phase quotas. | tests whether stale-data control alone is enough for the hard family. | may neither solve the hard family nor preserve healthy-block strength. |

## Interpretation

- The shortlist stays inside the existing teacher-guided KL learner-state family.
- The saturation program covers ten distinct mechanism directions with twenty bounded variants before narrowing, so a negative result would reflect a real family-level search inside the existing learner-state surface rather than another one-off tweak.
| `round5_phase_balanced_4096` | cleanup_round_plus_phase_balanced_replay | the strongest prior signal may need both one extra cleanup round and a phase-balanced recent buffer. | combines late cleanup extension with stale-data control on the hard-family dev split. | can over-specialize to dev blocks and erode healthy-block robustness. |

## Interpretation

- The shortlist stays inside the existing teacher-guided KL learner-state family.
- The saturation program covers ten distinct mechanism directions with twenty bounded variants before narrowing, so a negative result would reflect a real family-level search inside the existing learner-state surface rather than another one-off tweak.
| `round5_phase_balanced_6144` | cleanup_round_plus_phase_balanced_replay | the same mixed mechanism may need a wider recent buffer to carry enough context into hard-family cleanup. | tests whether the round-5 plus phase-balanced idea survives a less aggressive cap. | can give back the stale-data benefit while still paying the extra-round cost. |

## Interpretation

- The shortlist stays inside the existing teacher-guided KL learner-state family.
- The saturation program covers ten distinct mechanism directions with twenty bounded variants before narrowing, so a negative result would reflect a real family-level search inside the existing learner-state surface rather than another one-off tweak.
| `round5_phase_balanced_dis050` | cleanup_round_phase_balanced_disagreement | a modest disagreement bonus on top of the best mixed replay shape may focus the hard-family late mismatches more precisely. | combines stale-data control, extra cleanup, and disagreement targeting on the same hard-family slice. | may over-focus on disagreement-heavy failures and create brittle wins. |

## Interpretation

- The shortlist stays inside the existing teacher-guided KL learner-state family.
- The saturation program covers ten distinct mechanism directions with twenty bounded variants before narrowing, so a negative result would reflect a real family-level search inside the existing learner-state surface rather than another one-off tweak.
| `round5_phase_balanced_dis100` | cleanup_round_phase_balanced_disagreement | the strongest disagreement-targeted mixed variant may be required if the remaining hard-family errors are still sparse and late. | saturates the combined cleanup, replay, and disagreement hypothesis on the hard-family dev split. | can overshoot and turn the program into a late-phase overfit. |

## Interpretation

- The shortlist stays inside the existing teacher-guided KL learner-state family.
- The saturation program covers ten distinct mechanism directions with twenty bounded variants before narrowing, so a negative result would reflect a real family-level search inside the existing learner-state surface rather than another one-off tweak.
