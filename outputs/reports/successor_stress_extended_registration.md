# Successor Stress Registration

- frozen benchmark pack: `outputs/reports/frozen_benchmark_pack.json`
- current canonical successor pack: `outputs/reports/hard_family_saturation_successor_pack.json`
- current canonical successor: `round6`
- predecessor line: `post_unlock_weighted`
- git commit: `c40df2053b0d7631b557fbd9c46c8b9296d2f8f6`
- git dirty: `True`

## Prospective Blocks

- fresh stress blocks: `[{'lane': 'prospective_g', 'seeds': [251, 257, 263]}, {'lane': 'prospective_h', 'seeds': [269, 271, 277]}]`

## Candidate Lineup

- `post_unlock_weighted`: thaw-qualified predecessor line for context
- `round5`: one-round-shorter bracket challenger
- `round6`: current canonical successor line
- `round7`: one-more-round challenger
- `round10`: later-round challenger
- `round12`: far-later-round challenger

## Plan

- Stage 1: screen the current successor, predecessor, and later-round challengers on two new prospective DoorKey blocks.
- Stage 2: run matched structured controls only for `round6` and the best challenger from Stage 1.
- Stage 3: route- and stability-check the validated line before deciding whether round6 still holds the incumbent role.
