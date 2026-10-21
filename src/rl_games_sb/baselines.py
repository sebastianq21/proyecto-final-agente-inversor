"""Simple strategies the RL agent must beat. A policy maps an observation to an
action index of invest_env.ACTIONS (0 -> 0%, 1 -> 50%, 2 -> 100% invested).

Contents:
  - buy_and_hold() : 100% invested from the first day to the last
  - half()         : 50% invested, rebalanced daily
  - random_policy(): a uniformly random action every day, from a seeded RNG
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

Policy = Callable[[np.ndarray], int]


def buy_and_hold() -> Policy:
    return lambda obs: 2


def half() -> Policy:
    return lambda obs: 1


def random_policy(seed: int) -> Policy:
    rng = np.random.default_rng(seed)
    return lambda obs: int(rng.integers(3))
