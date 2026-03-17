# DoorKey Claim-Broadening Decision Memo

## Decision

The DoorKey teacher-guided `KL` learner-state `SARE` claim can now broaden within DoorKey, but it remains explicitly bounded:

- teacher-guided extraction only
- DoorKey only
- external `64`-episode evaluation only

The strongest clean statement supported by this repo is now:

- teacher-guided `KL` learner-state supervision gives multi-expert `SARE` a real DoorKey edge over the matched teacher-guided tokenized control
- that DoorKey edge survives the missing `single_expert` control on the original matched lane
- recovered DoorKey success remains causally routing-dependent under the bounded perturbation family, but not every probe is equally strong on every recovered seed

This is still not:

- a PPO-only routed win
- a cross-task routed advantage claim

## Answers

1. Does the DoorKey edge survive the missing matched `single_expert` control?

Yes, on the original matched lane.

The evidence is in [lss_single_expert_matched_control_report.md](lss_single_expert_matched_control_report.md):

- seed `7`: `KL` learner-state `single_expert` `1.0000`, `KL` learner-state `SARE` `1.0000`
- seed `11`: `KL` learner-state `single_expert` `1.0000`, `KL` learner-state `SARE` `0.5625`
- seed `19`: `KL` learner-state `single_expert` `0.0000`, `KL` learner-state `SARE` `0.5781`

Mean greedy success on that original matched lane:

- `KL` learner-state `token_dense`: `0.6667`
- `KL` learner-state `single_expert`: `0.6667`
- `KL` learner-state `SARE`: `0.7135`

So the missing fairness control does not erase the current DoorKey `SARE` edge.

2. Does causal route-dependence generalize beyond the original `7/23` demonstration?

Mostly yes, but with an important nuance.

The evidence is in [lss_extended_route_dependence_report.md](lss_extended_route_dependence_report.md):

- on recovered seeds `7`, `19`, and `23`, every single-expert ablation, fixed-router override, and route randomization drops greedy success to `0.0000`
- on recovered seed `29`, every single-expert ablation except one and the fixed-router override still collapse or severely damage success, but route randomization only drops greedy success from `1.0000` to `0.9844`

So routing dependence now extends beyond the original 2-seed demonstration, but the random-routing probe is not uniformly decisive across all recovered seeds.

3. Does one more fresh matched seed block strengthen, preserve, or weaken the edge?

Strengthen.

The evidence is in [lss_additional_fresh_seed_block_report.md](lss_additional_fresh_seed_block_report.md):

- seed `37`: `KL` learner-state `token_dense` `1.0000`, `KL` learner-state `SARE` `1.0000`
- seed `41`: `KL` learner-state `token_dense` `0.0000`, `KL` learner-state `SARE` `1.0000`
- seed `43`: `KL` learner-state `token_dense` `0.0000`, `KL` learner-state `SARE` `0.4688`

Mean greedy success on the extra fresh block:

- `KL` learner-state `token_dense`: `0.3333`
- `KL` learner-state `SARE`: `0.8229`

No new routed seed falls to `0.0`.

4. What is the right DoorKey claim now?

The right claim is now:

- a multi-expert routed DoorKey edge under teacher-guided `KL` learner-state extraction

with explicit scope limits:

- DoorKey only
- teacher-guided extraction only
- external `64`-episode evaluation only

The current combined DoorKey picture is in [lss_expanded_combined_doorkey_report.md](lss_expanded_combined_doorkey_report.md):

- `KL` learner-state `token_dense`: mean greedy success `0.5139`, complete-seed failures `4`
- `KL` learner-state `single_expert`: mean greedy success `0.6667` on the original 3-seed fairness slice, complete-seed failures `1`
- `KL` learner-state `SARE`: mean greedy success `0.8455`, complete-seed failures `0`

5. Should work continue within DoorKey, stay frozen, or pause?

Continue within DoorKey only.

That means:

- keep matched teacher-guided tokenized controls mandatory
- keep the `single_expert` fairness control visible in DoorKey summaries
- keep causal route-perturbation checks on any future successful routed checkpoints
- do not reopen PPO-only routed claims
- do not broaden beyond DoorKey while [lss_keycorridor_transfer_report.md](lss_keycorridor_transfer_report.md) stays flat

## Supporting Artifacts

- Reproduction note: [lss_claim_broadening_reproduction_note.md](lss_claim_broadening_reproduction_note.md)
- Missing-control fairness: [lss_single_expert_matched_control_report.md](lss_single_expert_matched_control_report.md)
- Extended route dependence: [lss_extended_route_dependence_report.md](lss_extended_route_dependence_report.md)
- Additional fresh block: [lss_additional_fresh_seed_block_report.md](lss_additional_fresh_seed_block_report.md)
- Expanded combined DoorKey report: [lss_expanded_combined_doorkey_report.md](lss_expanded_combined_doorkey_report.md)
