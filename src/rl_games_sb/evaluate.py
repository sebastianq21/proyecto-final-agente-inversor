"""Backtests: run agents and baselines over one period and measure them alike.

Contents:
  - run()        : one deterministic pass over a period -> a daily DataFrame
  - metrics()    : cumulative return, max drawdown, costs, trades, ...
  - ensemble()   : one policy from several: the median of their actions
  - backtest()   : RL agents (per seed, optionally their ensemble) + baselines
                   over a split; writes CSVs and figures under results/<split>/
  - plot_curves(): learning curves on train vs val, from results/curves/

Every strategy runs in the same env (googl-<split>-v0): same days, same
10 bps cost, same starting capital. Agents act greedily (deterministic=True).
"""
from __future__ import annotations

from itertools import cycle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from rl_games_sb import baselines, config, envs, registry
from rl_games_sb.invest_env import ACTIONS

RESULTS = Path("results")
TRADING_DAYS = 252

# Validated categorical palette (light mode), assigned in fixed order.
COLORS = {"agent": "#2a78d6", "buy_and_hold": "#eb6834", "half": "#1baf7a", "random": "#8a8984"}
AGENT_COLORS = ["#2a78d6", "#6250d6", "#e87ba4", "#008300", "#eda100", "#e34948"]


def run(env_id: str, policy: baselines.Policy) -> pd.DataFrame:
    """Day-by-day value, invested fraction (the action's target) and costs."""
    env = envs.make(env_id)
    obs, info = env.reset(seed=0)
    rows = [{"date": info["date"], "value": info["value"], "target": 0.0, "costs": 0.0}]
    done = False
    while not done:
        action = int(policy(obs))
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rows.append({
            "date": info["date"], "value": info["value"],
            "target": ACTIONS[action], "costs": info["costs"],
        })
    env.close()
    return pd.DataFrame(rows).set_index("date")


def metrics(day: pd.DataFrame) -> dict[str, float]:
    value = day["value"]
    daily = value.pct_change().dropna()
    targets = day["target"].iloc[1:]  # the actions taken; row 0 is the start
    return {
        "cum_return": value.iloc[-1] / value.iloc[0] - 1,
        "max_drawdown": (value / value.cummax() - 1).min(),
        "costs": day["costs"].iloc[-1],
        "trades": int((targets.diff().fillna(targets.iloc[0]) != 0).sum()),
        "exposure": targets.mean(),
        "ann_vol": daily.std() * np.sqrt(TRADING_DAYS),
        "sharpe": daily.mean() / daily.std() * np.sqrt(TRADING_DAYS) if daily.std() else 0.0,
    }


def _agent_policy(cfg: config.Config, seed: int) -> baselines.Policy:
    model = registry.load(cfg.algo, cfg.env, device="cpu", seed=seed, tag=cfg.save_tag)
    return lambda obs: model.predict(obs, deterministic=True)[0]


def ensemble(policies: list[baselines.Policy]) -> baselines.Policy:
    """Each day, the median of the seeds' actions. Actions are ordered
    (0 %, 50 %, 100 % invested), so the median is the position the majority
    is at least as cautious as, and it is always one of the actions."""
    def policy(obs):
        actions = sorted(int(p(obs)) for p in policies)
        return actions[len(actions) // 2]
    return policy


def backtest(
    split: str, configs: list[str], seeds: int = 5, random_runs: int = 100,
    with_ensemble: bool = False, out: Path | None = None,
) -> pd.DataFrame:
    """Per-run metrics for every agent seed and baseline, plus a mean row per
    strategy. Writes <out>/metrics.csv, summary.csv and figures; `out`
    defaults to results/<split>/. with_ensemble adds a "<config>_ensemble"
    strategy per config: one run of the median of its seeds."""
    env_id = f"googl-{split}-v0"
    out = out or RESULTS / split
    out.mkdir(parents=True, exist_ok=True)

    runs: dict[str, list[pd.DataFrame]] = {}
    for path in configs:
        cfg = config.load(path)
        name = cfg.path.stem
        policies = [_agent_policy(cfg, s) for s in range(seeds)]
        runs[name] = [run(env_id, p) for p in policies]
        if with_ensemble:
            runs[f"{name}_ensemble"] = [run(env_id, ensemble(policies))]
    runs["buy_and_hold"] = [run(env_id, baselines.buy_and_hold())]
    runs["half"] = [run(env_id, baselines.half())]
    runs["random"] = [run(env_id, baselines.random_policy(s)) for s in range(random_runs)]

    table = pd.DataFrame([
        {"strategy": name, "run": i, **metrics(day)}
        for name, days in runs.items() for i, day in enumerate(days)
    ])
    table.to_csv(out / "metrics.csv", index=False)
    summary = table.drop(columns="run").groupby("strategy", sort=False).agg(["mean", "std"])
    summary.to_csv(out / "summary.csv")

    for name, days in runs.items():
        pd.concat({i: d for i, d in enumerate(days)}, names=["run"]).to_csv(out / f"days_{name}.csv")
    _plot_equity(runs, out / "equity.png", split)
    for path in configs:
        name = Path(path).stem
        _plot_positions(runs[name], out / f"positions_{name}.png", name)
    return summary


def _style(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b5b4ae")
    ax.tick_params(colors="#52514e", labelsize=9)


LABELS = {  # display names in the figures (Spanish, the report's language)
    "buy_and_hold": "Comprar y mantener", "half": "50 % fijo", "random": "Aleatorio",
    "googl_dqn_sparse": "DQN R0", "googl_dqn_log": "DQN R1", "googl_dqn_dd": "DQN R2",
    "googl_dqn_turnover": "DQN R3", "googl_dqn_turnover_noise05": "DQN R3 + ruido 0,5",
    "googl_dqn_turnover_noise10": "DQN R3 + ruido 1,0",
    "googl_dqn_turnover_noise05_2m": "DQN 2M", "googl_ddqn_turnover_noise05_2m": "DDQN 2M",
    "googl_ddqn_dsr_noise05_2m": "DDQN 2M Sharpe dif.",
    "multi_ddqn_turnover_noise05_2m": "DDQN 2M multiactivo",
}


def _label(name: str) -> str:
    base = name.removesuffix("_ensemble")
    return LABELS.get(base, base) + (" (ensamble)" if base != name else "")


SPLIT_NAMES = {"train": "entrenamiento", "val": "validación", "test": "prueba"}


def _usd(x: float, _=None) -> str:
    return "USD " + f"{x:,.0f}".replace(",", ".")


def _plot_equity(runs: dict[str, list[pd.DataFrame]], path: Path, split: str) -> None:
    """Mean portfolio value per strategy; agents also show each seed, faintly.
    End-of-line labels are nudged apart so they never overlap."""
    fig, ax = plt.subplots(figsize=(10, 5))
    agent_colors = cycle(AGENT_COLORS)
    ends = []
    for name, days in runs.items():
        values = pd.concat([d["value"] for d in days], axis=1)
        color = COLORS.get(name) or next(agent_colors)
        if name not in COLORS:  # an RL agent: one faint line per seed
            ax.plot(values.index, values, color=color, linewidth=0.8, alpha=0.25)
        label = _label(name)
        legend = f"{label} (media de {len(days)})" if len(days) > 1 else label
        mean = values.mean(axis=1)
        ax.plot(mean.index, mean, color=color, linewidth=2, label=legend)
        ends.append([mean.iloc[-1], label])
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.045
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    x_end = max(d.index[-1] for days in runs.values() for d in days)
    for y, label in ends:
        ax.annotate(label, (x_end, y), xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color="#52514e", va="center")
    _style(ax)
    ax.set_title(f"Valor del portafolio, período de {SPLIT_NAMES[split]} (inicio: USD 100.000)",
                 loc="left", fontsize=11)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_usd))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_positions(days: list[pd.DataFrame], path: Path, name: str) -> None:
    """Invested fraction over time for each seed, one small panel per seed."""
    fig, axes = plt.subplots(len(days), 1, figsize=(10, 1.1 * len(days) + 0.6), sharex=True)
    for seed, (ax, day) in enumerate(zip(np.atleast_1d(axes), days)):
        ax.step(day.index, day["target"], where="post", color=COLORS["agent"], linewidth=1.2)
        _style(ax)
        ax.set_yticks(ACTIONS, [f"{a:.0%}" for a in ACTIONS])
        ax.set_ylabel(f"semilla {seed}", fontsize=8, color="#52514e")
    np.atleast_1d(axes)[0].set_title(
        f"Fracción invertida: {_label(name)}", loc="left", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(configs: list[str], name: str = "curves") -> Path:
    """Mean (over seeds) score vs. training steps, on train and on val, one
    line per config. Score = log of the growth of the capital over the whole
    period (0 = break even); buy and hold is the dashed reference."""
    splits = ("train", "val")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for stem, color in zip((Path(c).stem for c in configs), AGENT_COLORS):
        runs = [pd.read_csv(f) for f in sorted((RESULTS / "curves").glob(f"{stem}_seed*.csv"))]
        curve = pd.concat(runs).groupby("step").mean()
        for ax, split in zip(axes, splits):
            ax.plot(curve.index, curve[f"googl-{split}-v0"], color=color, linewidth=2,
                    label=f"{stem.removeprefix('googl_dqn_')} ({len(runs)} seeds)")
    for ax, split in zip(axes, splits):
        hold = np.log(run(f"googl-{split}-v0", baselines.buy_and_hold())["value"])
        ax.axhline(hold.iloc[-1] - hold.iloc[0], color=COLORS["buy_and_hold"], linewidth=1.5,
                   linestyle="--", label="buy and hold")
        ax.axhline(0, color="#b5b4ae", linewidth=0.8)
        _style(ax)
        ax.set_title(f"{split}: log(final / initial capital)", loc="left", fontsize=11)
        ax.set_xlabel("training steps", color="#52514e", fontsize=9)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = RESULTS / "curves" / f"{name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
