# Frozen Claim Envelope

- manifest: `configs/claims/doorkey_frozen_claim.yaml`
- git commit: `1604c1e30ff9bd6f07abe0f35c8fabb82bb63449`
- git dirty: `False`

## Allowed Claim

- `bounded teacher-guided DoorKey SARE result`

## Not Allowed

- `ppo_only_routed_win`
- `specifically_multi_expert_routed_advantage`
- `cross_task_routed_advantage`
- `keycorridor_transfer_claim`

## Future Thaw Bar

- Any future DoorKey retry must use the `external 64-episode policy_diagnostics path`.
- On seeds `[47, 53, 59]`, candidate KL learner-state `SARE` must beat the frozen retry-block mean `0.3125`.
- On that same block, candidate KL learner-state `SARE` must at least match the matched KL learner-state `single_expert` result.
- The candidate must not worsen the combined DoorKey KL learner-state `SARE` mean `0.7122`.

## Operational Rule

- No future DoorKey result should be treated as a thaw candidate until it passes the automated claim gate against this manifest.
- The sealed frozen benchmark pack and pack-based claim gate are the canonical operational entrypoints for future work.
