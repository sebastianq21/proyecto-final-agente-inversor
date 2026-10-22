"""Download the extra training assets from Yahoo Finance and describe them.

    uv run python scripts/download_assets.py

Only dates up to the end of the train period (2015-12-31) are downloaded, so
the extra assets cannot leak val or test information. Prices are adjusted for
splits and dividends (auto_adjust). Writes data/assets/<TICKER>.csv (same
columns as data/STOCKS_GOOGL.csv) and results/ampliacion/datos/: a summary
table per asset over the train period and a chart of normalized prices.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from rl_games_sb import data

TICKERS = ("AAPL", "MSFT", "AMZN", "INTC", "CSCO", "ORCL", "IBM", "QCOM", "ADBE", "NVDA", "SPY")
START = "2008-09-01"  # ~3 months of warm-up so the 50-day features exist from 2009
END = "2016-01-01"    # yfinance's end is exclusive: last day 2015-12-31
OUT = Path("results/ampliacion/datos")


def download(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)  # single ticker: drop the ticker level
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index.name = "Date"
    return df


def describe(name: str, df: pd.DataFrame, googl: pd.Series) -> dict:
    close = df["Close"].loc[slice(*data.SPLITS["train"])]
    ret = np.log(close).diff().dropna()
    years = len(ret) / 252
    return {
        "asset": name, "first": close.index[0].date(), "last": close.index[-1].date(),
        "days": len(close),
        "annual_return": (close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1,
        "annual_vol": ret.std() * np.sqrt(252),
        "max_drawdown": (close / close.cummax() - 1).min(),
        "corr_googl": ret.corr(googl.reindex(ret.index)),
    }


def main() -> None:
    data.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    googl = data.load()
    googl_ret = np.log(googl["Close"]).diff()
    assets = {"GOOGL": googl}
    for ticker in TICKERS:
        df = download(ticker)
        df.to_csv(data.ASSETS_DIR / f"{ticker}.csv")
        assets[ticker] = df
        print(f"{ticker}: {len(df)} days, {df.index[0].date()} .. {df.index[-1].date()}")

    table = pd.DataFrame([describe(n, df, googl_ret) for n, df in assets.items()])
    table.to_csv(OUT / "resumen.csv", index=False)
    print(table.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, df in assets.items():
        close = df["Close"].loc[slice(*data.SPLITS["train"])]
        ax.plot(close.index, close / close.iloc[0], linewidth=2.2 if name == "GOOGL" else 0.9,
                color="#2a78d6" if name == "GOOGL" else None, label=name)
    ax.set_yscale("log")
    ax.set_title("Precio normalizado (escala log), período de entrenamiento 2009-2015",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(OUT / "precios.png", dpi=150)


if __name__ == "__main__":
    main()
