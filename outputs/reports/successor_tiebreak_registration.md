# Successor Stress Registration

- frozen benchmark pack: `outputs/reports/frozen_benchmark_pack.json`
- current canonical successor pack: `outputs/reports/hard_family_saturation_successor_pack.json`
- current canonical successor: `round6`
- predecessor line: `post_unlock_weighted`
- git commit: `909d1d72796e0d6990d936db38a6a693ea0858fd`
- git dirty: `True`

## Prospective Blocks

- fresh stress blocks: `[{'lane': 'prospective_i', 'seeds': [281, 283, 293]}, {'lane': 'prospective_j', 'seeds': [307, 311, 313]}]`

## Candidate Lineup

- `post_unlock_weighted`: thaw-qualified predecessor line for context
- `round6`: current canonical successor line
- `round7`: direct tie-break challenger

## Plan

- Stage 1: screen the current successor, predecessor, and later-round challengers on two new prospective DoorKey blocks.
- Stage 2: run matched structured controls only for `round6` and the best challenger from Stage 1.
- Stage 3: route- and stability-check the validated line before deciding whether round6 still holds the incumbent role.
