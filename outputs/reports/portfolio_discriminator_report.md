# Portfolio Discriminator Report

- source campaign: `lss_portfolio_campaign`
- git commit: `65103a3f8aff38dce5908eaf6f0fcb7af734b9a0`
- git dirty: `True`
- verified tie-bucket candidates: `['round10', 'round10_post_unlock_x4_dis025', 'round10_post_unlock_x5', 'round10_carry2_post4', 'round10_conf_post4', 'round10_door2_post4']`
- representative classification line: `round10`

## Dev Seed Classification

- global-hard seeds: `['prospective_c/193']`
- shared-parity seeds: `['prospective_c/181', 'prospective_c/191', 'prospective_d/197', 'prospective_d/199', 'prospective_d/211', 'prospective_f/239', 'prospective_f/241']`
- route/control differentiator seeds: `['prospective_f/233']`

## Subset Means

| Line | Overall SARE | Overall token_dense | Overall single_expert | Global-Hard SARE | Parity SARE | Differentiator SARE | Differentiator token_dense | Differentiator single_expert |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `round6` | `0.8889` | `0.8472` | `0.8889` | `0.0000` | `1.0000` | `1.0000` | `0.6250` | `1.0000` |
| `round10` | `0.8889` | `0.8472` | `0.8889` | `0.0000` | `1.0000` | `1.0000` | `0.6250` | `1.0000` |
| `round10_post_unlock_x4_dis025` | `0.8889` | `0.8472` | `0.8889` | `0.0000` | `1.0000` | `1.0000` | `0.6250` | `1.0000` |
| `round10_post_unlock_x5` | `0.8889` | `0.8472` | `0.8889` | `0.0000` | `1.0000` | `1.0000` | `0.6250` | `1.0000` |
| `round10_carry2_post4` | `0.8889` | `0.8472` | `0.8889` | `0.0000` | `1.0000` | `1.0000` | `0.6250` | `1.0000` |
| `round10_conf_post4` | `0.8889` | `0.8472` | `0.8889` | `0.0000` | `1.0000` | `1.0000` | `0.6250` | `1.0000` |
| `round10_door2_post4` | `0.8889` | `0.8472` | `0.8889` | `0.0000` | `1.0000` | `1.0000` | `0.6250` | `1.0000` |

## Interpretation

- The portfolio dev split does not contain multiple challenger-discriminating seeds. It contains one global-hard sentinel, one route/control differentiator, and everything else is shared parity.
- On the only differentiator seed (`prospective_f/233`), `round6` and every verified challenger are still tied at `1.0000`; the only separation there is against matched `token_dense` at `0.6250`.
- That means the verified tie bucket remains a true tie even after removing the globally hard seed from view. The right operational conclusion is to keep `round6` as incumbent and stop spending search budget on variants that merely re-enter this bucket.
