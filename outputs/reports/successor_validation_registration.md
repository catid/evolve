# Successor Validation Registration

- frozen benchmark pack: `outputs/reports/frozen_benchmark_pack.json`
- current canonical successor pack: `outputs/reports/hard_family_saturation_successor_pack.json`
- current canonical successor: `round6`
- predecessor line: `post_unlock_weighted`
- git commit: `21a890285831feb89c6de03ea16619239ff732e3`
- git dirty: `True`

## Fresh Blocks

- fresh validation blocks: `[{'lane': 'prospective_a', 'seeds': [151, 157, 163]}, {'lane': 'prospective_b', 'seeds': [167, 173, 179]}]`

## Candidate Lineup

- `post_unlock_weighted`: canonical thaw-qualified predecessor line
- `round5`: one-extra-round successor ablation
- `round6`: canonization-qualified successor line
- `round7`: exploratory extra-round successor line

## Plan

- Stage 1: screen the successor line on genuinely fresh DoorKey blocks against the thaw-qualified predecessor.
- Stage 2: run matched structured controls for `round6` and `round7` on the same blocks.
- Stage 3: run one fresh route probe for the best validated line and write the decision memo.
