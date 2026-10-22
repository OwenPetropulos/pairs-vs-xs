"""
Performance metrics and "why did it break down" diagnostics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

LABELS = {
    "fund_pairs_ls": "Pairs L/S · fundamental",
    "fund_xs_ls": "Cross-sectional L/S · fundamental",
    "price_pairs_ls": "Pairs L/S · spread",
    "price_xs_ls": "Cross-sectional L/S · spread",
    "fund_pairs_lo": "Pairs long-only · fundamental",
    "fund_xs_lo": "Cross-sectional long-only · fundamental",
    "price_pairs_lo": "Pairs long-only · spread",
    "price_xs_lo": "Cross-sectional long-only · spread",
    "ew": "Equal-weight universe",
    "spy": "S&P 500 (SPY)",
}


def _daily(nav: pd.Series) -> pd.Series:
    return nav.pct_change().dropna()


def metrics(res: dict) -> pd.DataFrame:
    spy = _daily(res["spy"]["nav"])
    rows = []
    for k, lab in LABELS.items():
        r = res[k]
        nav = r["nav"]
        d = _daily(nav)
        yrs = (nav.index[-1] - nav.index[0]).days / 365.25
        cagr = nav.iloc[-1] ** (1 / yrs) - 1
        vol = d.std() * np.sqrt(252)
        sharpe = d.mean() * 252 / vol if vol > 0 else np.nan
        dn = d[d < 0].std() * np.sqrt(252)
        sortino = d.mean() * 252 / dn if dn > 0 else np.nan
        dd = (nav / nav.cummax() - 1).min()
        al = pd.concat([d, spy], axis=1, join="inner").dropna()
        beta = np.cov(al.iloc[:, 0], al.iloc[:, 1])[0, 1] / al.iloc[:, 1].var()
        corr = al.iloc[:, 0].corr(al.iloc[:, 1])
        m = nav.resample("ME").last().pct_change().dropna()
        rows.append({
            "key": k, "Strategy": lab,
            "CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Sortino": sortino,
            "Max DD": dd, "Calmar": cagr / abs(dd) if dd < 0 else np.nan,
            "Beta (SPY)": beta, "Corr (SPY)": corr,
            "Avg gross": r["gross"].mean(), "Turnover/yr": r["turnover_ann"],
            "% months up": (m > 0).mean(),
        })
    return pd.DataFrame(rows).set_index("key")


def subperiods(res: dict) -> pd.DataFrame:
    out = {}
    for k, lab in LABELS.items():
        d = _daily(res[k]["nav"])
        row = {}
        for s, e in C.SUBPERIODS:
            x = d.loc[s:e]
            if len(x) < 60:
                row[f"{s[:4]}–{min(int(e[:4]), d.index[-1].year)}"] = np.nan
                continue
            row[f"{s[:4]}–{min(int(e[:4]), d.index[-1].year)}"] = x.mean() / x.std() * np.sqrt(252)
        out[lab] = row
    return pd.DataFrame(out).T


def rolling_sharpe(res: dict, keys: list[str]) -> pd.DataFrame:
    out = {}
    w = C.ROLLING_MONTHS
    for k in keys:
        m = res[k]["nav"].resample("ME").last().pct_change()
        out[k] = m.rolling(w).mean() / m.rolling(w).std() * np.sqrt(12)
    return pd.DataFrame(out).dropna(how="all")


def strategy_correlations(res: dict) -> pd.DataFrame:
    keys = ["fund_pairs_ls", "fund_xs_ls", "price_pairs_ls", "price_xs_ls", "spy"]
    m = pd.DataFrame({LABELS[k]: res[k]["nav"].resample("ME").last().pct_change() for k in keys}).dropna()
    return m.corr()


def signal_ic(adj: pd.DataFrame, S: dict, dates: pd.DatetimeIndex,
              horizons=(1, 3, 6, 12)) -> pd.DataFrame:
    """
    Does cheapness predict relative returns at all, before any trading rule?
    Each month: rank correlation between the cheapness z-score and the forward
    k-month relative return (pairs: A minus B; cross-section: stock minus the
    universe average). Months are sampled every k so windows don't overlap,
    and the t-stat is mean / (sd / sqrt(n)).
    """
    px = adj.reindex(dates)
    rows = []
    for name, key in (("Pairs · fundamental", "fund_pairs"), ("Cross-section · fundamental", "fund_xs"),
                      ("Pairs · spread", "price_pairs"), ("Cross-section · spread", "price_xs")):
        z = S[key]
        for k in horizons:
            fwd = px.shift(-k) / px - 1
            if key.endswith("pairs"):
                rel = pd.DataFrame({p: fwd[p.split("/")[0]] - fwd[p.split("/")[1]] for p in z.columns})
            else:
                f = fwd[z.columns]
                rel = f.sub(f.mean(axis=1), axis=0)
            ics = []
            for d in z.index[::k]:
                a, b = z.loc[d], rel.loc[d] if d in rel.index else None
                if b is None:
                    continue
                m = a.notna() & b.notna()
                if m.sum() >= 6:
                    ics.append(a[m].rank().corr(b[m].rank()))
            ics = pd.Series(ics).dropna()
            rows.append({"Signal": name, "Horizon (m)": k, "Mean IC": ics.mean(),
                         "t-stat": ics.mean() / ics.std() * np.sqrt(len(ics)) if len(ics) > 2 else np.nan,
                         "% months IC>0": (ics > 0).mean(), "N": len(ics)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pair-level breakdown
# ---------------------------------------------------------------------------
def pair_attribution(res: dict, key: str) -> pd.DataFrame:
    """Cumulative contribution (sum of daily P&L / NAV) of each pair's two legs."""
    c = res[key]["contrib"]
    out = {}
    for a, b, _ in C.PAIRS:
        out[f"{a}/{b}"] = c[[a, b]].sum(axis=1).cumsum()
    return pd.DataFrame(out)


def trade_table(trades: list[dict], res: dict, key: str, z: pd.DataFrame) -> pd.DataFrame:
    c = res[key]["contrib"]
    rows = []
    for t in trades:
        a, b = t["pair"].split("/")
        win = c.loc[t["entry"]:t["exit"], [a, b]]
        pnl = float(win.values.sum()) if len(win) else 0.0
        future = z[t["pair"]].loc[t["entry"]:].iloc[1:C.CONVERGE_HORIZON + 1]
        converged = bool((future.abs() < C.CONVERGE_Z).any())
        months = max(1, round((t["exit"] - t["entry"]).days / 30.4))
        rows.append({**t, "pnl": pnl, "months": months, "converged_in_horizon": converged})
    return pd.DataFrame(rows)


def rolling_pair_corr(adj: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    r = adj.pct_change(fill_method=None)
    out = {}
    for a, b, _ in C.PAIRS:
        out[f"{a}/{b}"] = r[a].rolling(window, min_periods=int(window * 0.8)).corr(r[b])
    return pd.DataFrame(out).resample("ME").last()


def half_life(gap: pd.Series) -> float:
    """
    Mean-reversion half-life of a spread, in observations, from the AR(1)
    regression  d(gap)_t = a + b * gap_{t-1}.  b >= 0 means no pull back to
    the mean at all, reported as inf. A half-life that blows up from the first
    half of the sample to the second is the cleanest sign a pair has 'broken'.
    """
    g = gap.dropna()
    if len(g) < 30:
        return np.nan
    x = g.shift(1).iloc[1:].values
    y = g.diff().iloc[1:].values
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0][1]
    return float(np.log(2) / -b) if b < 0 else np.inf


def pair_summary(res: dict, trades_tbl: pd.DataFrame, key: str, z: pd.DataFrame,
                 corr: pd.DataFrame, gaps: dict) -> pd.DataFrame:
    """gaps: pair -> raw (un-z-scored) gap series used by this signal."""
    attr = pair_attribution(res, key)
    start = attr.index[0]
    mid = start + (attr.index[-1] - start) / 2
    rows = []
    for p in attr.columns:
        tt = trades_tbl[trades_tbl["pair"] == p] if len(trades_tbl) else trades_tbl
        cc = corr[p].loc[start:]
        zz = z[p].dropna()
        g = gaps[p].loc[start:]
        a_mid = attr[p].loc[:mid]
        rows.append({
            "Pair": p,
            "Total contrib": attr[p].iloc[-1],
            "Contrib 1st half": a_mid.iloc[-1] if len(a_mid) else np.nan,
            "Contrib 2nd half": attr[p].iloc[-1] - (a_mid.iloc[-1] if len(a_mid) else 0.0),
            "Half-life 1st (m)": half_life(g.loc[:mid]),
            "Half-life 2nd (m)": half_life(g.loc[mid:]),
            "Trades": len(tt),
            "Hit rate": (tt["pnl"] > 0).mean() if len(tt) else np.nan,
            "Converged ≤6m": tt["converged_in_horizon"].mean() if len(tt) else np.nan,
            "Stop-outs": int((tt["exit_reason"] == "stop").sum()) if len(tt) else 0,
            "Stop-out P&L": tt.loc[tt["exit_reason"] == "stop", "pnl"].sum() if len(tt) else 0.0,
            "Avg hold (m)": tt["months"].mean() if len(tt) else np.nan,
            "% months |z|>stop": (zz.abs() > C.STOP_Z).mean() if len(zz) else np.nan,
            "Corr 1st half": cc.loc[:mid].mean(),
            "Corr 2nd half": cc.loc[mid:].mean(),
        })
    return pd.DataFrame(rows).set_index("Pair").sort_values("Total contrib")


def findings(M: pd.DataFrame, summaries: dict, tables: dict, sub: pd.DataFrame) -> list[str]:
    """Plain-language facts generated from the numbers, as a starting point for the writeup."""
    out = []
    for src, name in (("fund", "fundamental"), ("price", "spread")):
        sp, sx = M.loc[f"{src}_pairs_ls", "Sharpe"], M.loc[f"{src}_xs_ls", "Sharpe"]
        better = "pairs" if sp > sx else "cross-sectional"
        out.append(
            f"On the {name} signal, pairs L/S Sharpe was {sp:.2f} vs {sx:.2f} for cross-sectional "
            f"L/S — the {better} version did better."
        )
        s = summaries[src]
        worst = s.head(3)
        out.append(
            f"Weakest {name} pairs by contribution: "
            + ", ".join(f"{p} ({v:+.1%})" for p, v in worst["Total contrib"].items()) + "."
        )
        t = tables[src]
        if len(t):
            stops = t[t["exit_reason"] == "stop"]
            conv = t["converged_in_horizon"].mean()
            out.append(
                f"{name.capitalize()} pairs: {len(t)} trades, {conv:.0%} converged within "
                f"{C.CONVERGE_HORIZON} months; {len(stops)} stop-outs cost {stops['pnl'].sum():+.1%} "
                f"vs {t.loc[t['exit_reason'] != 'stop', 'pnl'].sum():+.1%} from all other trades."
            )
        lo_p, lo_x, ew = (M.loc[f"{src}_pairs_lo", "CAGR"], M.loc[f"{src}_xs_lo", "CAGR"], M.loc["ew", "CAGR"])
        out.append(
            f"Long-only {name}: cheap-sibling portfolio CAGR {lo_p:.1%}, cross-sectional cheapest-third "
            f"{lo_x:.1%}, vs {ew:.1%} for equal weight."
        )
    lab = {"fund": LABELS["fund_pairs_ls"], "price": LABELS["price_pairs_ls"]}
    for src in ("fund", "price"):
        row = sub.loc[lab[src]].dropna()
        if len(row):
            out.append(
                f"{lab[src]}: best sub-period {row.idxmax()} (Sharpe {row.max():.2f}), "
                f"worst {row.idxmin()} (Sharpe {row.min():.2f})."
            )
    return out
