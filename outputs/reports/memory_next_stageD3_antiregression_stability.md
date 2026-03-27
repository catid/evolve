# Memory Next Stage D3 Anti-Regression Stability

- stage-d3 survivors: `7`

| Candidate | Healthy Gap | Stability Greedy | Option Duration | Option Switch | Decision |
| --- | --- | --- | --- | --- | --- |
| partial225_gap_t055 | 0.4219 | 0.5469 | 1.2929 | 0.4253 | advance |
| partial225_greedy | 0.4219 | 0.5469 | 1.2930 | 0.4263 | advance |
| select_partial225_greedy | 0.4219 | 0.5469 | 1.2930 | 0.4263 | advance |
| partial22_conf_gate085_t055 | 0.4219 | 0.5469 | 2.3983 | 0.1986 | advance |
| partial22_conf_gate092_t055 | 0.4219 | 0.5469 | 2.3443 | 0.2011 | advance |
| partial22_cons2_t05 | 0.4219 | 0.5469 | 2.3003 | 0.2104 | advance |
| partial22_cons2_t055 | 0.4219 | 0.5469 | 2.3003 | 0.2104 | advance |

## Interpretation

- Stage D3 checks that a candidate is not winning by damaging the rest of the Memory branch.
- Surviving candidates must keep healthy-band behavior and remain stable across the dedicated stability seed groups.
