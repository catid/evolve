# Portfolio Round-Prior Dossier

- source campaign: `lss_portfolio_campaign`
- git commit: `89117bb4c70e9d06c521ef1367654eb694f5a26a`
- git dirty: `True`
- active incumbent portfolio dev mean: `0.8889`

## Round Prior Comparison

| Candidate | Portfolio Dev Mean | Delta vs round6 | Delta vs token | Delta vs single | Mean Aggregate Steps | Mean Fine-Tune Steps | c/193 | f/233 | h/269 | h/277 | Bucket |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `round5` | `0.8368` | `-0.0521` | `-0.0104` | `-0.0521` | `74232.8` | `2320.9` | `0.0000` | `1.0000` | `1.0000` | `1.0000` | `cheapest_but_below_incumbent` |
| `round7` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `79841.9` | `2496.4` | `0.0000` | `1.0000` | `1.0000` | `1.0000` | `recommended_conservative_default` |
| `round10` | `0.8889` | `0.0000` | `0.0417` | `0.0000` | `86907.2` | `2717.8` | `0.0000` | `1.0000` | `1.0000` | `1.0000` | `higher_cost_same_signal` |

## Search Default

- recommended conservative default: `['round7']`
- cheaper but below-incumbent priors: `['round5']`
- higher-cost same-signal priors: `['round10']`

## Interpretation

- On the measured triage seeds, `round5`, `round7`, and `round10` are identical: all fail `prospective_c/193`, all preserve `prospective_f/233 = 1.0000`, and all solve `prospective_h/269` and `prospective_h/277` at `1.0000`.
- The difference is outside that narrow surface. `round5` is cheaper, but it is also below the incumbent on the broader portfolio dev mean, so it should not be the default conservative restart.
- `round7` and `round10` tie the incumbent on the broader portfolio dev mean, but `round7` does so at lower aggregate and fine-tune cost than `round10`.
- That makes `round7` the clean conservative default if future bounded search restarts from the measured round-count priors.
