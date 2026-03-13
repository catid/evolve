#!/usr/bin/env bash
set -euo pipefail

milestone1=$(bd create "Milestone 1: Bootstrap + MiniGrid PPO baseline" -p 0 -t epic --silent)
milestone2=$(bd create "Milestone 2: Routed variants on MiniGrid" -p 1 -t epic --silent)
milestone3=$(bd create "Milestone 3: Procgen + analysis" -p 2 -t epic --silent)

bootstrap=$(bd create "Bootstrap repo, environment, and Beads workflow" -p 0 -t task --parent "$milestone1" --silent)
dense=$(bd create "Implement tokenized dense RL baseline" -p 0 -t task --parent "$milestone1" --silent)
single=$(bd create "Implement tokenized single-expert baseline" -p 1 -t task --parent "$milestone1" --silent)
sare=$(bd create "Implement SARE routed core" -p 0 -t task --parent "$milestone2" --silent)
tregh=$(bd create "Implement TREG-H routed core" -p 1 -t task --parent "$milestone2" --silent)
srw=$(bd create "Implement SRW routed relational core" -p 1 -t task --parent "$milestone2" --silent)
por=$(bd create "Implement POR option-routing core" -p 1 -t task --parent "$milestone2" --silent)
ppo=$(bd create "Implement PPO + DDP training stack" -p 0 -t task --parent "$milestone1" --silent)
minigrid=$(bd create "Integrate MiniGrid experiments" -p 0 -t task --parent "$milestone1" --silent)
procgen=$(bd create "Integrate Procgen experiments" -p 2 -t task --parent "$milestone3" --silent)
metrics=$(bd create "Add metrics, logging, and analysis tooling" -p 0 -t task --parent "$milestone3" --silent)
tests=$(bd create "Write smoke tests and reproducibility checks" -p 0 -t task --parent "$milestone3" --silent)
ablations=$(bd create "Run baseline and routed ablations" -p 0 -t task --parent "$milestone3" --silent)
readme=$(bd create "Write README and experiment guide" -p 1 -t task --parent "$milestone3" --silent)

bd dep add "$dense" "$bootstrap"
bd dep add "$dense" "$ppo"
bd dep add "$dense" "$minigrid"
bd dep add "$single" "$dense"
bd dep add "$sare" "$dense"
bd dep add "$tregh" "$sare"
bd dep add "$srw" "$sare"
bd dep add "$por" "$sare"
bd dep add "$procgen" "$dense"
bd dep add "$procgen" "$sare"
bd dep add "$metrics" "$dense"
bd dep add "$metrics" "$sare"
bd dep add "$tests" "$dense"
bd dep add "$tests" "$sare"
bd dep add "$ablations" "$single"
bd dep add "$ablations" "$sare"
bd dep add "$ablations" "$tregh"
bd dep add "$ablations" "$srw"
bd dep add "$ablations" "$por"
bd dep add "$ablations" "$metrics"
bd dep add "$readme" "$dense"
bd dep add "$readme" "$tests"

cat <<EOF
Seeded Beads graph:
  milestone1=$milestone1
  milestone2=$milestone2
  milestone3=$milestone3
  bootstrap=$bootstrap
  dense=$dense
  single=$single
  sare=$sare
  tregh=$tregh
  srw=$srw
  por=$por
  ppo=$ppo
  minigrid=$minigrid
  procgen=$procgen
  metrics=$metrics
  tests=$tests
  ablations=$ablations
  readme=$readme
EOF
