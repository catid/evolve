# Post-PASS Campaign Registration

- frozen manifest: `configs/claims/doorkey_frozen_claim.yaml`
- frozen benchmark pack: `outputs/reports/frozen_benchmark_pack.json`
- current thaw-qualified candidate pack: `outputs/reports/long_campaign_candidate_pack.json`
- current candidate: `post_unlock_weighted`
- git commit: `90f4ca1e3b9a572156e49d4af86d273a748cea43`
- git dirty: `True`

## New Seed Blocks

- `post_pass_a`: `[61, 67, 71]`
- `post_pass_b`: `[73, 79, 83]`

## Fairness And Mechanism Plan

- Stage 1: run the thaw-qualified `post_unlock_weighted` candidate plus matched post_unlock_weighted structured controls on both fresh blocks.
- Stage 2: aggregate fairness across the historical campaign blocks and both new fresh blocks.
- Stage 3: extend route validation to two additional historically strong recovered seeds and two historically weak retry-block seeds.
- Stage 4: evaluate multiple candidate checkpoints on one weak representative and one stronger representative, with frozen-baseline stability side-by-side.
- Stage 5: draft a successor benchmark pack without replacing the frozen pack.
- Stage 6: package the expanded candidate, run the frozen-pack gate, and decide canonization vs thaw-only vs fallback.

## Canonization Bar

- The candidate must remain ahead of matched structured controls on both new fresh blocks.
- The expanded DoorKey picture must preserve or improve the current combined candidate mean while staying inside the DoorKey-only teacher-guided envelope.
- Route disruption must remain materially harmful on both added strong seeds and historically weak seeds.
- The selected candidate checkpoints must not reduce to narrow one-checkpoint spikes.
- The expanded candidate pack must still clear the existing frozen-pack gate.
