# Claim-Hardening Decision Memo

## Decision

The reopened teacher-guided DoorKey `SARE` claim is now stronger, but it should still stay narrowly scoped.

## Answers

1. Does KL learner-state `SARE` remain positive on additional seeds?

Yes. On fresh DoorKey seeds `23`, `29`, and `31`, KL learner-state `SARE` reached greedy success `1.0000`, `1.0000`, and `1.0000` under the external 64-episode evaluation path. See [lss_additional_seed_report.md](lss_additional_seed_report.md).

2. How does it compare against matched teacher-guided tokenized controls?

Teacher-guided KL learner-state supervision helps `token_dense` too, improving its original-seed DoorKey mean from `0.5677` to `0.6667`. But KL learner-state `SARE` still keeps the higher mean at `0.7135` on the same `7/11/19` lane. See [lss_matched_control_report.md](lss_matched_control_report.md).

3. Is the reopened claim now robust enough to broaden, or still narrow?

It is robust enough to stay alive on DoorKey, but not broad enough to generalize beyond a narrow extraction-method claim. The additional-seed replication strengthens DoorKey specifically, while the matched-control result keeps the story method-first rather than routing-first. See [lss_additional_seed_report.md](lss_additional_seed_report.md) and [lss_matched_control_report.md](lss_matched_control_report.md).

4. Does any evidence justify moving to KeyCorridor?

No. The bounded KeyCorridor transfer check is flat: recovered `token_dense`, baseline PPO `SARE`, and KL learner-state `SARE` all stay at greedy success `0.0000` on seeds `7`, `11`, and `19`. See [lss_keycorridor_transfer_report.md](lss_keycorridor_transfer_report.md).

5. Should routed work in this repo continue, pause, or stay narrowly scoped?

Stay narrowly scoped. The current positive claim is:

- DoorKey only
- teacher-guided KL learner-state extraction only
- external 64-episode evaluation only

It is not a PPO-only routed win and not a cross-task routed advantage claim.

## Route Integrity

The strengthened DoorKey result still looks routed on a newly recovered seed. On seed `23`, KL learner-state `SARE` moves greedy success from `0.0000` to `1.0000` while route entropy stays near the PPO baseline (`1.3857 -> 1.3804`) and active compute stays fixed at `0.5000`. See [lss_new_case_route_integrity_report.md](lss_new_case_route_integrity_report.md).
