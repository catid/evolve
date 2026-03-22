# Portfolio Frontier Guard Workflow Contract

- git commit: `251c92bc33d7b9ec5aa313dcbdd1a9c245bfa7fe`
- git dirty: `True`
- overall: `pass`
- workflow name: `portfolio-frontier-guard`
- guard script: `scripts/run_portfolio_frontier_guard.sh`
- syntax check command: `bash -n scripts/run_portfolio_frontier_guard.sh`
- guard run command: `bash ./scripts/run_portfolio_frontier_guard.sh`
- upload artifact name: `portfolio-frontier-guard-reports`
- workflow dispatch enabled: `True`

## Upload Paths

- `outputs/reports/portfolio_frontier_docs_audit.md`
- `outputs/reports/portfolio_frontier_docs_audit.json`
- `outputs/reports/portfolio_frontier_doctor.md`
- `outputs/reports/portfolio_frontier_doctor.json`
- `outputs/reports/portfolio_frontier_guard_report.md`
- `outputs/reports/portfolio_frontier_guard_report.json`
- `outputs/reports/portfolio_frontier_guard_workflow_contract.md`
- `outputs/reports/portfolio_frontier_guard_workflow_contract.json`
- `outputs/reports/portfolio_active_state_doctor.md`
- `outputs/reports/portfolio_active_state_doctor.json`
- `outputs/reports/portfolio_operational_state.md`
- `outputs/reports/portfolio_operational_state.json`
- `outputs/reports/portfolio_seed_pack_doctor.md`
- `outputs/reports/portfolio_seed_pack_doctor.json`

## Checks

| Check | Status | Detail |
| --- | --- | --- |
| `workflow_name_matches` | `pass` | workflow_name=portfolio-frontier-guard |
| `workflow_dispatch_enabled` | `pass` | workflow_dispatch_enabled=True |
| `syntax_check_present` | `pass` | required_command=bash -n scripts/run_portfolio_frontier_guard.sh |
| `guard_run_present` | `pass` | required_command=bash ./scripts/run_portfolio_frontier_guard.sh |
| `upload_step_present` | `pass` | uses=actions/upload-artifact@v4 |
| `upload_artifact_name_matches` | `pass` | upload_artifact_name=portfolio-frontier-guard-reports |
| `upload_paths_cover_required` | `pass` | required_upload_count=14 |
