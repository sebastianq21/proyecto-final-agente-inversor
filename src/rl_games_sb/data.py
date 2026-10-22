"""Market data for the investing env: GOOGL daily prices, features and splits.

Contents:
  - DATA_PATH  : the dataset (a copy of gym-anytrading's STOCKS_GOOGL)
  - ASSETS_DIR : extra train-only assets for the multi-asset env
  - SPLITS     : date ranges of the train / val / test periods
  - FEATURES   : names of the state features, in order
  - load()     : read the price CSV
  - features() : price indicators at each date, using only data up to that date
  - split()    : one period's prices and normalized features

Lookahead rules:
  - every feature at date t is a backward-looking window ending at t
    (rolling(...) and shift(+k) only, never shift(-k));
  - normalization uses the mean / std of the train period only;
  - a period's first rows may use prices from the period before it (e.g. the
    50-day trend on 2017-01-03 reads late 2016). That is information already
    known at t, not future information.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "STOCKS_GOOGL.csv"
# extra training assets (scripts/download_assets.py), dates up to 2015 only
ASSETS_DIR = DATA_PATH.parent / "assets"

SPLITS = {
    "train": ("2009-01-01", "2015-12-31"),
    "val": ("2016-01-01", "2016-12-31"),
    "test": ("2017-01-01", "2018-12-31"),  # held-out: data ends 2018-08-29
}

FEATURES = ("ret_1", "ret_5", "ret_20", "gap_sma20", "trend_20_50", "vol_20", "rsi_14", "volume_z")


def load(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI, scaled from [0, 100] to [-0.5, 0.5]."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + gain / loss)) / 100 - 0.5


def features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per date with the FEATURES columns. Rows without a full
    50-day history are dropped."""
    close, log_close = df["Close"], np.log(df["Close"])
    sma20, sma50 = close.rolling(20).mean(), close.rolling(50).mean()
    ret_1 = log_close.diff()
    out = pd.DataFrame({
        "ret_1": ret_1,                                    # last day's move
        "ret_5": log_close.diff(5),                        # 1-week momentum
        "ret_20": log_close.diff(20),                      # 1-month momentum
        "gap_sma20": close / sma20 - 1,                    # stretch vs. short trend
        "trend_20_50": sma20 / sma50 - 1,                  # short vs. medium trend
        "vol_20": ret_1.rolling(20).std(),                 # risk regime
        "rsi_14": _rsi(close),                             # overbought / oversold
        "volume_z": np.log(df["Volume"] / df["Volume"].rolling(20).mean()),  # unusual activity
    })
    return out.dropna()


def split(name: str, df: pd.DataFrame | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """Close prices and z-scored features for one period.

    The z-score uses the train period's mean and std, whatever `name` is.
    """
    df = load() if df is None else df
    feats = features(df)
    start, end = SPLITS["train"]
    train = feats.loc[start:end]
    feats = (feats - train.mean()) / train.std()

    start, end = SPLITS[name]
    feats = feats.loc[start:end]
    return df["Close"].loc[feats.index], feats
