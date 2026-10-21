#!/usr/bin/env bash
# Reward iteration: train each reward variant with 5 seeds, then compare them
# on the validation period (2016). Never touches the test period.
#   bash scripts/train_rewards.sh [variant ...]     (default: sparse log dd turnover)
# A variant names configs/googl_dqn_<variant>.yaml, e.g. turnover_noise05.
set -euo pipefail
REWARDS=${*:-sparse log dd turnover}
SEEDS="0 1 2 3 4"
for r in $REWARDS; do
  env=$(awk '/^env:/ {print $2}' "configs/googl_dqn_$r.yaml")
  for s in $SEEDS; do
    uv run rlgames-sb delete dqn --env "$env" --seed "$s"   # start fresh, never resume
    uv run rlgames-sb train --config "configs/googl_dqn_$r.yaml" --seed "$s"
  done
done
configs=""
for r in $REWARDS; do configs="$configs configs/googl_dqn_$r.yaml"; done
uv run rlgames-sb backtest $configs --split val
