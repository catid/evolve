# Canonization Stage 1 Mechanism Shortlist

- blocker: `post_pass_b` stays token-dense-led even after the thaw-qualified `post_unlock_weighted` intervention
- hard-block family for this campaign: `post_pass_b=[73, 79, 83], post_pass_c=[89, 97, 101]`

| Candidate | Mechanism Hypothesis | Hard-Block Target | How It Could Fail | Strong-Block Risk |
| --- | --- | --- | --- | --- |
| `post_unlock_weighted_x6` | a stronger post-unlock KL weight can close the remaining token_dense gap on hard late-phase blocks without touching the routed architecture | `['post_pass_b', 'post_pass_c']` | overweights late clean-up states and regresses the already healthy strong routed blocks | may flatten route usage by forcing a narrow late-phase action pattern |
| `post_unlock_weighted_disagreement075` | a modest disagreement bonus on top of post-unlock weighting can target the hard-block states where routed SARE still disagrees with the teacher late in the episode | `['post_pass_b', 'post_pass_c']` | amplifies noisy late disagreements and helps all structured students equally | may destabilize checkpoint selection by chasing disagreement-heavy rounds |
| `post_unlock_weighted_round5` | one extra learner-state clean-up round can convert near-miss hard-block late episodes without changing the architecture or extraction family | `['post_pass_b', 'post_pass_c']` | overfits late rounds and weakens the stronger blocks through a narrow checkpoint spike | can produce brittle one-round gains that fail stability checks |

## Interpretation

- All shortlisted interventions stay inside the existing teacher-guided KL learner-state family.
- The shortlist is deliberately bounded to post-unlock weighting, disagreement-aware weighting, and one cleanup-round extension rather than a broad knob search.
