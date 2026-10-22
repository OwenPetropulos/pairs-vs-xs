"""
Signals.

Every signal is a "cheapness" score: positive = cheap relative to the reference.
The pairs and cross-sectional versions are built the same way; the ONLY thing
that differs is the reference each stock is measured against.

  Pairs, fundamental : gap = yield_A - yield_B                 (reference: sibling)
  XS,    fundamental : gap = yield_i - median(universe yields) (reference: universe)
  Pairs, spread      : gap = log P_A - log P_B
  XS,    spread      : gap = log P_i - mean(log P universe)

In each case the gap is z-scored against its own trailing history, so a pair
that has always traded at a different multiple (KO vs PEP) is not mistaken for
an opportunity. For yields, a high z means cheap; for prices, a high z means
rich, so the price z is negated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each month."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)


def _zscore(x: pd.DataFrame | pd.Series, window: int, min_periods: int):
    mu = x.rolling(window, min_periods=min_periods).mean()
    sd = x.rolling(window, min_periods=min_periods).std()
    return (x - mu) / sd.replace(0, np.nan)


def pair_metrics(yields: dict) -> dict:
    """Choose the metric per pair, falling back to E/P when EBIT/EV coverage is thin."""
    chosen = {}
    start = pd.Timestamp(C.BACKTEST_START)
    for a, b, m in C.PAIRS:
        if m == "ev_ebit":
            y = yields["ev_ebit"].loc[start:, [a, b]]
            cov = y.notna().mean().min()
            if cov < C.MIN_COVERAGE:
                print(f"  {a}/{b}: EBIT/EV coverage {cov:.0%} -> falling back to E/P")
                m = "ep"
        chosen[(a, b)] = m
    return chosen


# ---------------------------------------------------------------------------
# Fundamental
# ---------------------------------------------------------------------------
def fund_pairs(yields: dict, metrics: dict) -> pd.DataFrame:
    out = {}
    for (a, b), m in metrics.items():
        y = yields[m]
        gap = y[a] - y[b]
        out[f"{a}/{b}"] = _zscore(gap, C.FUND_Z_WINDOW, C.FUND_Z_MIN)
    return pd.DataFrame(out)


def fund_xs(yields: dict, metrics: dict) -> pd.DataFrame:
    stock_metric = {}
    for (a, b), m in metrics.items():
        stock_metric[a] = m
        stock_metric[b] = m
    med = {m: yields[m][list(stock_metric)].median(axis=1) for m in set(stock_metric.values())}
    out = {}
    for t, m in stock_metric.items():
        gap = yields[m][t] - med[m]
        out[t] = _zscore(gap, C.FUND_Z_WINDOW, C.FUND_Z_MIN)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Spread (price)
# ---------------------------------------------------------------------------
def price_pairs(adj: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    logp = np.log(adj)
    out = {}
    w = C.PRICE_Z_WINDOW
    for a, b, _ in C.PAIRS:
        gap = logp[a] - logp[b]
        out[f"{a}/{b}"] = -_zscore(gap, w, int(w * 0.8))
    return pd.DataFrame(out).reindex(dates)


def price_xs(adj: pd.DataFrame, dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    logp = np.log(adj[tickers])
    ref = logp.mean(axis=1)  # equal-weight geometric basket
    gap = logp.sub(ref, axis=0)
    w = C.PRICE_Z_WINDOW
    return (-_zscore(gap, w, int(w * 0.8))).reindex(dates)


def build_all(prices, yields: dict, tickers: list[str]) -> dict:
    dates = rebalance_dates(prices.adj.index)
    dates = dates[dates >= pd.Timestamp(C.BACKTEST_START) - pd.offsets.MonthEnd(1)]
    # Yields are month-stamped; align them to the last trading day of each month.
    # History before BACKTEST_START is kept for the z-score warm-up.
    all_dates = rebalance_dates(prices.adj.index)
    y_full = {}
    for m, df in yields.items():
        df = df.copy()
        df.index = df.index.to_period("M")
        tmp = df.reindex(all_dates.to_period("M"))
        tmp.index = all_dates
        y_full[m] = tmp

    metrics = pair_metrics(y_full)
    logp = np.log(prices.adj)
    gaps = {
        "fund": pd.DataFrame({f"{a}/{b}": y_full[m][a] - y_full[m][b] for (a, b), m in metrics.items()}),
        "price": pd.DataFrame({f"{a}/{b}": logp[a] - logp[b] for a, b, _ in C.PAIRS}).reindex(all_dates),
    }
    sig = {
        "fund_pairs": fund_pairs(y_full, metrics).reindex(dates),
        "fund_xs": fund_xs(y_full, metrics).reindex(dates),
        "price_pairs": price_pairs(prices.adj, dates),
        "price_xs": price_xs(prices.adj, dates, tickers),
    }
    return {"dates": dates, "signals": sig, "metrics": metrics, "gaps": gaps}
