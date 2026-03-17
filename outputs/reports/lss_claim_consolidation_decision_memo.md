# DoorKey Claim-Consolidation Decision Memo

## Decision

The DoorKey teacher-guided `SARE` claim is now stronger.

The strongest clean statement supported by this repo is:

- teacher-guided `KL` learner-state supervision gives `SARE` a real DoorKey edge over matched teacher-guided `token_dense`
- that edge is still bounded:
  - DoorKey only
  - teacher-guided extraction only
  - external `64`-episode evaluation only
- it is not a PPO-only routed win
- it is not a cross-task routed advantage claim

## Answers

1. Does the routed DoorKey edge survive fresh matched teacher-guided controls?

Yes.

Fresh matched DoorKey results are in [lss_fresh_matched_control_report.md](lss_fresh_matched_control_report.md):

- seed `23`: `KL` learner-state `token_dense` `0.0000`, `KL` learner-state `SARE` `1.0000`
- seed `29`: `KL` learner-state `token_dense` `0.6250`, `KL` learner-state `SARE` `1.0000`
- seed `31`: `KL` learner-state `token_dense` `1.0000`, `KL` learner-state `SARE` `1.0000`

Mean greedy success on the fresh matched lane:

- `KL` learner-state `token_dense`: `0.5417`
- `KL` learner-state `SARE`: `1.0000`

The combined original+fresh six-seed DoorKey view is in [lss_combined_doorkey_report.md](lss_combined_doorkey_report.md):

- `KL` learner-state `token_dense`: mean greedy success `0.6042`, complete-seed failures `2`
- `KL` learner-state `SARE`: mean greedy success `0.8568`, complete-seed failures `0`

2. Does recovered `SARE` performance actually depend on routing choices?

Yes.

The bounded causal probe family is in [lss_causal_route_dependence_report.md](lss_causal_route_dependence_report.md).

On recovered original seed `7` and recovered fresh seed `23`:

- every single-expert ablation drops greedy success from `1.0000` to `0.0000`
- fixed-router override drops greedy success from `1.0000` to `0.0000`
- route randomization drops greedy success from `1.0000` to `0.0000`

So the current positive result is not just “routing statistics stayed non-collapsed.” Under these probes, recovered DoorKey performance is causally routing-dependent.

3. Is the right claim now a routed DoorKey edge, an extraction-method win only, or still narrow / ambiguous?

The right claim is now:

- a routed DoorKey edge under teacher-guided `KL` learner-state extraction

but with explicit scope limits:

- teacher-guided extraction only
- DoorKey only
- external `64`-episode evaluation only

The claim is no longer just “extraction helps structured students generally,” because matched teacher-guided `token_dense` is included and still trails `SARE` on both the fresh matched lane and the combined six-seed DoorKey view.

4. Should work stay narrowly scoped on DoorKey, broaden within DoorKey, or pause again?

Broaden within DoorKey only.

That means:

- keep the teacher-guided extraction framing explicit
- keep matched teacher-guided tokenized controls as mandatory comparators
- keep causal route-dependence checks on successful routed checkpoints
- do not reopen PPO-only routed claims
- do not broaden to cross-task routed claims while KeyCorridor remains flat in [lss_keycorridor_transfer_report.md](lss_keycorridor_transfer_report.md)

## Supporting Artifacts

- Reproduction note: [lss_claim_consolidation_reproduction_note.md](lss_claim_consolidation_reproduction_note.md)
- Fresh matched controls: [lss_fresh_matched_control_report.md](lss_fresh_matched_control_report.md)
- Combined DoorKey report: [lss_combined_doorkey_report.md](lss_combined_doorkey_report.md)
- Causal route dependence: [lss_causal_route_dependence_report.md](lss_causal_route_dependence_report.md)
- Negative transfer check: [lss_keycorridor_transfer_report.md](lss_keycorridor_transfer_report.md)
