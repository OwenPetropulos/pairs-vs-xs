"""
Portfolio construction and the daily simulation engine.

Strategies (each run on both the fundamental and the spread signal):
  pairs_ls : within each pair, long the cheap sibling / short the rich one when
             |z| >= ENTRY_Z; hold until |z| < EXIT_Z; flat (and locked out until
             the gap resets) if |z| > STOP_Z. Each pair has a fixed 1/N slot.
  pairs_lo : long-only; hold the cheap sibling of every pair, 1/N each.
  xs_ls    : rank every stock vs the universe; long the cheapest third,
             short the richest third, each leg 100% of capital.
  xs_lo    : long-only; hold the cheapest third, equal weight.
Benchmarks:
  ew       : equal weight across the whole universe, rebalanced monthly.
  spy      : buy and hold SPY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# Target weights at each rebalance date
# ---------------------------------------------------------------------------
def pairs_ls(z: pd.DataFrame, tickers: list[str]):
    """Returns (weights DataFrame, trade log list)."""
    n = len(C.PAIRS)
    W = pd.DataFrame(0.0, index=z.index, columns=tickers)
    state = {p: 0 for p in z.columns}          # +1 long A/short B, -1 reverse, 0 flat
    locked = {p: False for p in z.columns}     # after a stop-out
    open_trade = {}
    trades = []

    for d in z.index:
        for p in z.columns:
            a, b = p.split("/")
            v = z.at[d, p]
            pos = state[p]
            if np.isnan(v):
                new = 0
                reason = "no signal"
            else:
                av = abs(v)
                if locked[p]:
                    if av < C.EXIT_Z:
                        locked[p] = False
                    new, reason = 0, None
                elif pos != 0:
                    if av > C.STOP_Z:
                        new, reason = 0, "stop"
                        locked[p] = True
                    elif av < C.EXIT_Z or np.sign(v) != pos:
                        new, reason = 0, "converged"
                    else:
                        new, reason = pos, None
                else:
                    if C.ENTRY_Z <= av <= C.STOP_Z:
                        new, reason = int(np.sign(v)), None
                    else:
                        new, reason = 0, None

            if pos != 0 and new != pos:
                t = open_trade.pop(p)
                t.update(exit=d, exit_reason=reason or "flip", exit_z=v)
                trades.append(t)
            if new != 0 and new != pos:
                open_trade[p] = dict(pair=p, side=new, entry=d, entry_z=v)
            state[p] = new
            if new != 0:
                W.at[d, a] += new / n
                W.at[d, b] -= new / n

    last = z.index[-1]
    for p, t in open_trade.items():
        t.update(exit=last, exit_reason="open at end", exit_z=z.at[last, p])
        trades.append(t)
    return W, trades


def pairs_lo(z: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    n = len(C.PAIRS)
    W = pd.DataFrame(0.0, index=z.index, columns=tickers)
    for p in z.columns:
        a, b = p.split("/")
        v = z[p]
        W[a] += np.where(v > 0, 1.0, np.where(v < 0, 0.0, 0.5)) / n
        W[b] += np.where(v < 0, 1.0, np.where(v > 0, 0.0, 0.5)) / n
    return W


def _xs_legs(z: pd.DataFrame, tickers: list[str], long_only: bool) -> pd.DataFrame:
    W = pd.DataFrame(0.0, index=z.index, columns=tickers)
    for d in z.index:
        row = z.loc[d].dropna()
        if len(row) < 6:
            continue
        k = max(1, int(round(len(row) * C.XS_QUANTILE)))
        ranked = row.sort_values()
        cheap = ranked.index[-k:]
        rich = ranked.index[:k]
        W.loc[d, cheap] = 1.0 / k
        if not long_only:
            W.loc[d, rich] = -1.0 / k
    return W


def xs_ls(z, tickers):
    return _xs_legs(z, tickers, long_only=False)


def xs_lo(z, tickers):
    return _xs_legs(z, tickers, long_only=True)


def equal_weight(dates, tickers, adj):
    W = pd.DataFrame(0.0, index=dates, columns=tickers)
    for d in dates:
        live = [t for t in tickers if not np.isnan(adj.at[d, t])]
        W.loc[d, live] = 1.0 / len(live)
    return W


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def simulate(W: pd.DataFrame, adj: pd.DataFrame, start=None, end=None) -> dict:
    """
    Daily NAV simulation. Weights decided at date d are traded at the close
    EXEC_LAG_DAYS trading days later and earn returns from the following day.
    Holdings drift with prices between rebalances.
    """
    cols = list(W.columns)
    rets = adj[cols].pct_change(fill_method=None).fillna(0.0)
    idx = rets.index
    first = idx.searchsorted(W.index[0])
    idx = idx[first:]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]

    # map each signal date to its execution date
    exec_map = {}
    for d in W.index:
        pos = rets.index.searchsorted(d) + C.EXEC_LAG_DAYS
        if pos < len(rets.index):
            exec_map[rets.index[pos]] = W.loc[d].values

    R = rets.loc[idx].values
    nav = 1.0
    h = np.zeros(len(cols))
    navs = np.empty(len(idx))
    contrib = np.zeros((len(idx), len(cols)))
    gross = np.zeros(len(idx))
    turnover = 0.0
    tc = C.TC_BPS / 1e4
    borrow = C.BORROW_BPS / 1e4 / 252

    for i, d in enumerate(idx):
        r = R[i]
        pnl = h * r
        contrib[i] = pnl / nav if nav else 0.0
        short_notional = -h[h < 0].sum()
        nav = nav + pnl.sum() - short_notional * borrow
        h = h * (1 + r)
        if d in exec_map:
            target = exec_map[d] * nav
            traded = np.abs(target - h).sum()
            turnover += traded / nav
            nav -= traded * tc
            h = exec_map[d] * nav
        navs[i] = nav
        gross[i] = np.abs(h).sum() / nav if nav else 0.0

    nav_s = pd.Series(navs, index=idx)
    years = (idx[-1] - idx[0]).days / 365.25
    return {
        "nav": nav_s,
        "contrib": pd.DataFrame(contrib, index=idx, columns=cols),
        "gross": pd.Series(gross, index=idx),
        "turnover_ann": turnover / years if years > 0 else np.nan,
    }


def run_all(prices, sig: dict, tickers: list[str]) -> dict:
    dates = sig["dates"]
    S = sig["signals"]
    adj = prices.adj
    results, trades = {}, {}

    for src in ("fund", "price"):
        W, tl = pairs_ls(S[f"{src}_pairs"], tickers)
        results[f"{src}_pairs_ls"] = simulate(W, adj)
        results[f"{src}_pairs_ls"]["weights"] = W
        trades[src] = tl
        results[f"{src}_pairs_lo"] = simulate(pairs_lo(S[f"{src}_pairs"], tickers), adj)
        results[f"{src}_xs_ls"] = simulate(xs_ls(S[f"{src}_xs"], tickers), adj)
        results[f"{src}_xs_lo"] = simulate(xs_lo(S[f"{src}_xs"], tickers), adj)

    adj_m = adj.reindex(dates)
    results["ew"] = simulate(equal_weight(dates, tickers, adj_m), adj)
    spy_w = pd.DataFrame({C.BENCHMARK: 1.0}, index=dates[:1])
    results["spy"] = simulate(spy_w, adj)

    # align every NAV to a common start (the first day all strategies exist)
    common = max(r["nav"].index[0] for r in results.values())
    for r in results.values():
        r["nav"] = r["nav"].loc[common:] / r["nav"].loc[common:].iloc[0]
        r["contrib"] = r["contrib"].loc[common:]
        r["gross"] = r["gross"].loc[common:]
    return {"results": results, "trades": trades}
