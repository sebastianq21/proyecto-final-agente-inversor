"""Checks that the investing env does not see the future and charges costs right.

Run: uv run pytest
"""
import numpy as np
import pandas as pd

from rl_games_sb import data
from rl_games_sb.invest_env import InvestEnv

CUT = pd.Timestamp("2014-06-02")


def _future_changed(df: pd.DataFrame, after: pd.Timestamp) -> pd.DataFrame:
    """A copy where every price and volume after `after` is scrambled."""
    df = df.astype(float)
    rows = df.index > after
    rng = np.random.default_rng(0)
    df.loc[rows, ["Close", "Volume"]] *= rng.uniform(0.5, 1.5, size=(rows.sum(), 2))
    return df


def test_features_ignore_the_future():
    df = data.load()
    before = data.features(df).loc[:CUT]
    after = data.features(_future_changed(df, CUT)).loc[:CUT]
    pd.testing.assert_frame_equal(before, after)


def test_normalization_uses_train_only():
    df = data.load()
    test_start = pd.Timestamp(data.SPLITS["test"][0])
    for name in ("train", "val"):
        _, before = data.split(name, df)
        _, after = data.split(name, _future_changed(df, test_start))
        pd.testing.assert_frame_equal(before, after)


def test_splits_do_not_overlap():
    idx = {name: data.split(name)[1].index for name in data.SPLITS}
    assert idx["train"].max() < idx["val"].min()
    assert idx["val"].max() < idx["test"].min()


def _run(env: InvestEnv, action: int) -> dict:
    env.reset(seed=0)
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info


def test_all_in_equals_buy_and_hold_minus_one_cost():
    env = InvestEnv("test", episode_len=None, cost_bps=10)
    info = _run(env, action=2)
    growth = env.price[-1] / env.price[0]
    assert np.isclose(info["value"], 100_000 * (1 - 0.001) * growth)
    assert np.isclose(info["costs"], 100.0)


def test_cash_stays_cash():
    info = _run(InvestEnv("test", episode_len=None), action=0)
    assert info["value"] == 100_000 and info["costs"] == 0


def test_log_rewards_add_up_to_growth():
    env = InvestEnv("val", episode_len=None, reward="log")
    env.reset(seed=0)
    total, done = 0.0, False
    while not done:
        _, reward, _, done, info = env.step(1)
        total += reward
    assert np.isclose(total, np.log(info["value"] / 100_000))


def test_decision_uses_the_next_close():
    """Investing at t earns the move from close t to close t+1, not before."""
    env = InvestEnv("val", episode_len=None, cost_bps=0)
    env.reset(seed=0)
    _, reward, *_ = env.step(2)
    assert np.isclose(reward, np.log(env.price[1] / env.price[0]))


def test_turnover_reward_charges_trading_only():
    """R3: same as the log reward, minus the penalty when the position changes."""
    log_env = InvestEnv("val", episode_len=None, reward="log")
    turn_env = InvestEnv("val", episode_len=None, reward="turnover", turnover_penalty=0.005)
    log_env.reset(seed=0)
    turn_env.reset(seed=0)
    steps = [(2, 0.005), (2, 0.0), (0, 0.005)]  # buy all, hold, sell all
    for action, penalty in steps:
        _, r_log, *_ = log_env.step(action)
        _, r_turn, *_ = turn_env.step(action)
        assert np.isclose(r_turn, r_log - penalty)


def test_obs_noise_hides_features_but_not_position_or_prices():
    clean = InvestEnv("val", episode_len=None, reward="turnover")
    noisy = InvestEnv("val", episode_len=None, reward="turnover", obs_noise=1.0)
    obs_c, _ = clean.reset(seed=0)
    obs_n, _ = noisy.reset(seed=0)
    assert not np.allclose(obs_c[:-1], obs_n[:-1])       # the features are perturbed
    assert obs_c[-1] == obs_n[-1]                         # the position is not
    for action in (2, 1, 0):
        _, r_c, *_ = clean.step(action)
        obs_n, r_n, *_ = noisy.step(action)
        assert np.isclose(r_c, r_n)                       # same prices, same rewards
    assert obs_n[-1] == clean.weight
