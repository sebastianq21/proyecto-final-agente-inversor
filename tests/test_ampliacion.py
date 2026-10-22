"""Extension study: tagged saves, seed ensemble and the differential Sharpe reward."""
from pathlib import Path

import numpy as np
import pandas as pd

from rl_games_sb import config, evaluate, registry
from rl_games_sb.invest_env import InvestEnv


def test_save_tag_names_the_file_and_default_is_unchanged():
    assert registry.save_path("dqn", "googl-log-v0", 0) == Path("saves/dqn_googl-log-v0_seed0.zip")
    assert registry.save_path("dqn", "googl-log-v0", 0, tag="2m") == Path(
        "saves/dqn_googl-log-v0_2m_seed0.zip"
    )


def test_config_reads_save_tag(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("env: googl-log-v0\nalgo: ddqn\ntimesteps: 10\nsave_tag: 2m\n")
    assert config.load(path).save_tag == "2m"
    path.write_text("env: googl-log-v0\nalgo: ddqn\ntimesteps: 10\n")
    assert config.load(path).save_tag == ""


def test_ensemble_takes_the_median_action():
    policies = [lambda obs, a=a: a for a in (0, 2, 2, 1, 0)]
    assert evaluate.ensemble(policies)(None) == 1
    policies = [lambda obs, a=a: a for a in (2, 2, 2, 0, 1)]
    assert evaluate.ensemble(policies)(None) == 2


def _rewards(env: InvestEnv, action: int, n: int = 30) -> list[float]:
    env.reset(seed=0)
    return [env.step(action)[1] for _ in range(n)]


def test_dsr_pays_nothing_in_cash():
    assert _rewards(InvestEnv(reward="dsr"), action=0) == [0.0] * 30


def test_dsr_first_step_has_the_sign_of_the_return():
    env = InvestEnv(reward="dsr", cost_bps=0)
    env.reset(seed=0)
    t = env.t
    reward = env.step(2)[1]
    move = env.price[t + 1] / env.price[t] - 1
    assert np.sign(reward) == np.sign(move) and reward != 0


def test_extra_assets_stop_before_val():
    from rl_games_sb import data
    val_start = pd.Timestamp(data.SPLITS["val"][0])
    for path in data.ASSETS_DIR.glob("*.csv"):
        assert data.load(path).index.max() < val_start, path


def test_multi_asset_env_samples_googl_and_the_extra_assets():
    env = InvestEnv(reward="turnover", assets=("AAPL", "SPY"))
    seen = set()
    for seed in range(30):
        env.reset(seed=seed)
        seen.add(env.asset)
    assert seen == {"GOOGL", "AAPL", "SPY"}
    obs, *_ = env.step(2)
    assert obs.shape == env.observation_space.shape
