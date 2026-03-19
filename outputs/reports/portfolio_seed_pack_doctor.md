# Portfolio Seed-Pack Doctor

- git commit: `d5a4513ed8d8e256b4ae6e7fbb0f906b7c498283`
- git dirty: `True`
- overall: `pass`

## Checks

| Check | Status | Detail |
| --- | --- | --- |
| `active_candidate_round6` | `pass` | seed_pack.active_candidate=round6, contract.active_candidate=round6 |
| `active_pack_matches_contract` | `pass` | seed_pack.active_candidate_pack=outputs/reports/portfolio_candidate_pack.json, contract.active_candidate_pack=outputs/reports/portfolio_candidate_pack.json |
| `archived_pack_matches_contract` | `pass` | seed_pack.archived_frozen_pack=outputs/reports/frozen_benchmark_pack.json, contract.archived_frozen_pack=outputs/reports/frozen_benchmark_pack.json |
| `restart_default_round7` | `pass` | restart_default=('round7',) |
| `reserve_priors_match_contract_roles` | `pass` | reserve_priors=('round10', 'round5') |
| `retired_priors_match_contract_roles` | `pass` | retired_priors=('door3_post5', 'post_unlock_x5') |
| `support_status_matches_contract` | `pass` | support_status=measured_support_regression |
| `generated_from_active_pack_current` | `pass` | generated_from.active_candidate_pack=outputs/reports/portfolio_candidate_pack.json |
| `seed_roles_match_contract_thresholds` | `pass` | support=SeedRole(lane='prospective_f', seed=233, required_min_success=1.0, required_behavior=None), weakness=SeedRole(lane='prospective_h', seed=269, required_min_success=0.984375001, required_behavior=None), guardrail=SeedRole(lane='prospective_h', seed=277, required_min_success=1.0, required_behavior=None), sentinel=SeedRole(lane='prospective_c', seed=193, required_min_success=None, required_behavior='track_only') |
| `validation_buckets_match_frontier_roles` | `pass` | validated_restart_default=['round7'], validated_reserve=['round10', 'round5'], validated_retired=['door3_post5', 'post_unlock_x5'], needs_review=[] |
| `scorer_verdicts_match_frontier_roles` | `pass` | scorer_grouped={'advance_for_broader_dev': ('round7', 'round10'), 'hold_seed_clean_but_below_incumbent': ('round5',), 'prune_guardrail_regression': ('post_unlock_x5',), 'prune_support_regression': ('door3_post5',)} |
