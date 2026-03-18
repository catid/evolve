# Successor Carryover Challenge Registration

- frozen benchmark pack: `outputs/reports/frozen_benchmark_pack.json`
- current canonical successor pack: `outputs/reports/hard_family_saturation_successor_pack.json`
- current canonical successor: `round6`
- git commit: `21a890285831feb89c6de03ea16619239ff732e3`
- git dirty: `True`

## Blocks

- hard-family development blocks: `[{'lane': 'post_pass_b', 'seeds': [73, 79, 83]}, {'lane': 'post_pass_c', 'seeds': [89, 97, 101]}, {'lane': 'post_pass_f', 'seeds': [137, 139, 149]}]`
- hard-family holdout blocks: `[{'lane': 'fresh_final', 'seeds': [47, 53, 59]}, {'lane': 'post_pass_e', 'seeds': [113, 127, 131]}]`
- healthy carryover blocks: `[{'lane': 'fresh', 'seeds': [23, 29, 31]}]`

## Challengers

- `round7`: one-more-round carryover challenger
- `round8`: two-more-round exploratory challenger

## Plan

- Stage 1: run challenger SARE lines across hard-family carryover blocks and one healthy carryover block.
- Stage 2: add matched token_dense and single_expert controls on the hard-family carryover split.
- Stage 3: route- and stability-check the best challenger before deciding whether it displaces round6.
