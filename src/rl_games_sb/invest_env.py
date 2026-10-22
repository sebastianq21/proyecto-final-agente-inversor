"""InvestEnv: invest a fraction of the capital in one asset (GOOGL), day by day.

Contents:
  - ACTIONS    : action index -> fraction of capital invested
  - REWARDS    : the reward variants (see InvestEnv._reward)
  - InvestEnv  : the Gymnasium env
  - ENV_IDS    : the registered ids, e.g. "googl-log-v0" (train) or "googl-test-v0"

Timing of one step (decision at the close of day t):
  1. the agent sees the features of day t (data up to the close of t) and its
     current invested fraction;
  2. it picks a target fraction; the portfolio is rebalanced at the close of t
     and pays `cost_bps` on the traded value;
  3. the price moves from close t to close t+1 and the portfolio value with it;
  4. the reward is computed from the values before (pre-cost) and after.
The agent never sees the close of t+1 before deciding.
"""
from __future__ import annotations

from typing import ClassVar

import gymnasium as gym
import numpy as np

from rl_games_sb import data

ACTIONS = (0.0, 0.5, 1.0)
REWARDS = ("log", "dd", "sparse", "turnover", "dsr")


class InvestEnv(gym.Env):
    """Train episodes are random `episode_len`-day windows of the period;
    with episode_len=None an episode is the whole period, from its first day.

    obs_noise > 0 adds Gaussian noise (std in z-score units) to the features
    the agent sees, not to its position nor to the prices: the same day never
    looks exactly the same twice, which makes memorizing it harder. Only
    training envs use it.

    assets (train only) adds extra tickers from data.ASSETS_DIR: each reset
    picks GOOGL or one of them uniformly, then a random window of it. Each
    asset is z-scored with its own train statistics. More market history to
    learn from, so less room to memorize GOOGL's.

    reward:
      - "log"    : log(V_{t+1} / V_t), costs included
      - "dd"     : "log" minus dd_penalty x the increase in drawdown
      - "turnover": "log" minus turnover_penalty x the traded fraction
                   |target - weight|, on top of the real cost: trading must
                   promise more than it costs
      - "sparse" : like gym-anytrading's stocks-v0: the dollar price gain of
                   the part of the position that is sold, paid only when
                   selling, and blind to costs
      - "dsr"    : differential Sharpe ratio (Moody & Saffell, 1998) of the
                   log return: how much this day's return raises an
                   exponential moving Sharpe ratio (rate dsr_eta). Rewards
                   return per unit of risk instead of raw return.
    """

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self, split: str = "train", reward: str = "log", episode_len: int | None = 252,
        cost_bps: float = 10.0, dd_penalty: float = 1.0, turnover_penalty: float = 0.005,
        obs_noise: float = 0.0, initial_value: float = 100_000.0, dsr_eta: float = 1 / 252,
        render_mode: str | None = None,  # accepted for envs.make(); nothing to render
        assets: tuple[str, ...] = (),
    ):
        assert reward in REWARDS, reward
        assert not assets or split == "train", "extra assets only exist up to 2015"
        self.series = {"GOOGL": data.split(split)} | {
            a: data.split(split, data.load(data.ASSETS_DIR / f"{a}.csv")) for a in assets
        }
        self._use("GOOGL")
        self.reward_kind, self.episode_len = reward, episode_len
        self.cost_rate, self.dd_penalty = cost_bps / 10_000, dd_penalty
        self.turnover_penalty, self.obs_noise = turnover_penalty, obs_noise
        self.initial_value, self.dsr_eta = initial_value, dsr_eta

        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        # the features plus the currently invested fraction
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(self.feats.shape[1] + 1,), dtype=np.float32
        )

    def _use(self, asset: str) -> None:
        self.asset = asset
        self.prices, feats = self.series[asset]
        self.dates = feats.index
        self.feats = feats.to_numpy(np.float32)
        self.price = self.prices.to_numpy(np.float64)

    def _obs(self) -> np.ndarray:
        feats = self.feats[self.t]
        if self.obs_noise:
            feats = feats + self.np_random.normal(0, self.obs_noise, feats.shape).astype(np.float32)
        return np.append(feats, np.float32(self.weight))

    def _info(self) -> dict:
        return {
            "date": self.dates[self.t], "value": self.value, "weight": self.weight,
            "costs": self.costs, "price": self.price[self.t],
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if len(self.series) > 1:
            self._use(list(self.series)[int(self.np_random.integers(len(self.series)))])
        last = len(self.price) - 1
        if self.episode_len is None:
            self.t, self.end = 0, last
        else:
            self.t = int(self.np_random.integers(0, last - self.episode_len + 1))
            self.end = self.t + self.episode_len
        self.value, self.peak, self.weight = self.initial_value, self.initial_value, 0.0
        self.costs, self.entry_price = 0.0, 0.0
        # moving 1st and 2nd moments of the return, for "dsr". B starts at a
        # (1 %)^2 daily variance so the first steps do not divide by zero.
        self.dsr_a, self.dsr_b = 0.0, 1e-4
        return self._obs(), self._info()

    def step(self, action):
        target = ACTIONS[int(action)]
        before = self.value

        # rebalance at the close of t
        traded = abs(target - self.weight)
        cost = self.cost_rate * traded * self.value
        sold = max(self.weight - target, 0.0)
        if target > self.weight:  # average entry price of the invested part
            self.entry_price = (
                self.weight * self.entry_price + (target - self.weight) * self.price[self.t]
            ) / target
        self.value -= cost
        self.costs += cost

        # the price moves from t to t+1; the invested fraction drifts with it
        r = self.price[self.t + 1] / self.price[self.t] - 1
        sale_gain = sold * (self.price[self.t] - self.entry_price)
        self.value *= 1 + target * r
        self.weight = target * (1 + r) / (1 + target * r)
        self.t += 1

        old_dd = 1 - before / self.peak
        self.peak = max(self.peak, self.value)
        new_dd = 1 - self.value / self.peak
        reward = self._reward(before, old_dd, new_dd, sale_gain, traded)

        truncated = self.t == self.end
        return self._obs(), reward, False, truncated, self._info()

    def _reward(
        self, before: float, old_dd: float, new_dd: float, sale_gain: float, traded: float
    ) -> float:
        if self.reward_kind == "sparse":
            return float(sale_gain)
        log_ret = float(np.log(self.value / before))
        if self.reward_kind == "dd":
            return log_ret - self.dd_penalty * max(new_dd - old_dd, 0.0)
        if self.reward_kind == "turnover":
            return log_ret - self.turnover_penalty * traded
        if self.reward_kind == "dsr":
            return self._dsr(log_ret)
        return log_ret

    def _dsr(self, r: float) -> float:
        """D_t = (B dA - A dB / 2) / (B - A^2)^(3/2), with A, B the moving means
        of r and r^2 before this step (Moody & Saffell, 1998). dA and dB already
        include eta, so this is the paper's D_t times eta: a constant scale."""
        a, b = self.dsr_a, self.dsr_b
        da, db = self.dsr_eta * (r - a), self.dsr_eta * (r * r - b)
        self.dsr_a, self.dsr_b = a + da, b + db
        var = b - a * a
        return float((b * da - 0.5 * a * db) / var**1.5) if var > 1e-12 else 0.0


# Training ids, one per reward; evaluation ids score every agent the same way
# (log return, so an episode's total reward is the log of its growth factor).
ENV_IDS = {f"googl-{r}-v0": {"split": "train", "reward": r} for r in REWARDS} | {
    # iteration 3: R3 with noisy observations (std 0.5 and 1.0)
    f"googl-turnover-noise{int(n * 10):02d}-v0": {"split": "train", "reward": "turnover", "obs_noise": n}
    for n in (0.5, 1.0)
} | {
    # extension: differential Sharpe reward, noise 0.5 like the group's final agent
    "googl-dsr-noise05-v0": {"split": "train", "reward": "dsr", "obs_noise": 0.5},
    # extension: R3 + noise 0.5 trained on GOOGL plus 11 more assets (2009-2015)
    "multi-turnover-noise05-v0": {
        "split": "train", "reward": "turnover", "obs_noise": 0.5,
        "assets": ("AAPL", "MSFT", "AMZN", "INTC", "CSCO", "ORCL", "IBM", "QCOM", "ADBE", "NVDA", "SPY"),
    },
} | {
    f"googl-{s}-v0": {"split": s, "reward": "log", "episode_len": None} for s in ("train", "val", "test")
}
for _id, _kwargs in ENV_IDS.items():
    gym.register(_id, entry_point=InvestEnv, kwargs=_kwargs)
