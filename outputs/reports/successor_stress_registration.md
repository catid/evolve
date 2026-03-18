# Successor Stress Registration

- frozen benchmark pack: `outputs/reports/frozen_benchmark_pack.json`
- current canonical successor pack: `outputs/reports/hard_family_saturation_successor_pack.json`
- current canonical successor: `round6`
- predecessor line: `post_unlock_weighted`
- git commit: `08252b4af2213d2109fb923a74103bdf61d6e614`
- git dirty: `True`

## Prospective Blocks

- fresh stress blocks: `[{'lane': 'prospective_c', 'seeds': [181, 191, 193]}, {'lane': 'prospective_d', 'seeds': [197, 199, 211]}]`

## Candidate Lineup

- `post_unlock_weighted`: thaw-qualified predecessor line for context
- `round6`: current canonical successor line
- `round7`: one-more-round challenger
- `round8`: two-more-round challenger
- `round9`: three-more-round challenger

## Plan

- Stage 1: screen the current successor, predecessor, and later-round challengers on two new prospective DoorKey blocks.
- Stage 2: run matched structured controls only for `round6` and the best challenger from Stage 1.
- Stage 3: route- and stability-check the validated line before deciding whether round6 still holds the incumbent role.
