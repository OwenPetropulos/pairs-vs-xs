"""
Data layer.

Prices   : yfinance daily bars (split-adjusted Close for market cap, Adj Close for returns).
Fundamentals : SEC EDGAR companyfacts, rebuilt point-in-time.

Point-in-time rule: a number is usable on date D only if the filing that first
reported it has a `filed` date <= D. Restatements in later filings are ignored
(we keep the earliest-filed value for each fiscal period), so the backtest never
sees a figure before the market did.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def universe() -> list[str]:
    out = []
    for a, b, _ in C.PAIRS:
        for t in (a, b):
            if t not in out:
                out.append(t)
    return out


def _cache_path(name: str) -> str:
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    return os.path.join(C.CACHE_DIR, name)


@dataclass
class PriceData:
    adj: pd.DataFrame      # dividend + split adjusted close (for returns)
    close: pd.DataFrame    # split adjusted close only (for market cap)
    splits: dict           # ticker -> pd.Series of split ratios indexed by date

    def splits_after(self, ticker: str, date: pd.Timestamp) -> float:
        s = self.splits.get(ticker)
        if s is None or len(s) == 0:
            return 1.0
        s = s[s.index > date]
        return float(np.prod(s.values)) if len(s) else 1.0


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
def load_prices(refresh: bool = False) -> PriceData:
    path = _cache_path("prices.pkl")
    if os.path.exists(path) and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)

    import yfinance as yf

    tickers = universe() + [C.BENCHMARK]
    raw = yf.download(
        tickers,
        start=C.DATA_START,
        end=C.BACKTEST_END,
        auto_adjust=False,
        actions=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    adj = raw["Adj Close"][tickers].sort_index()
    close = raw["Close"][tickers].sort_index()
    splits = {}
    if "Stock Splits" in raw.columns.get_level_values(0):
        sp = raw["Stock Splits"]
        for t in tickers:
            if t in sp.columns:
                s = sp[t]
                s = s[(s.notna()) & (s != 0) & (s != 1)]
                splits[t] = s.astype(float)

    missing = [t for t in tickers if adj[t].dropna().empty]
    if missing:
        raise RuntimeError(f"No price data for: {missing}")

    pdata = PriceData(adj=adj, close=close, splits=splits)
    with open(path, "wb") as f:
        pickle.dump(pdata, f)
    return pdata


# ---------------------------------------------------------------------------
# EDGAR
# ---------------------------------------------------------------------------
def _sec_get(url: str) -> dict:
    import requests

    if "your_email" in C.SEC_USER_AGENT:
        raise RuntimeError(
            "Set SEC_USER_AGENT in pvx/config.py to your name and email "
            "(SEC requires it) before fetching fundamentals."
        )
    r = requests.get(url, headers={"User-Agent": C.SEC_USER_AGENT}, timeout=30)
    r.raise_for_status()
    time.sleep(0.15)  # stay well under SEC's 10 req/s limit
    return r.json()


def _cik_map() -> dict:
    path = _cache_path("company_tickers.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    else:
        data = _sec_get("https://www.sec.gov/files/company_tickers.json")
        with open(path, "w") as f:
            json.dump(data, f)
    return {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}


def _merge_facts(a: dict, b: dict) -> dict:
    """Union of two companyfacts payloads (same company filed under two CIKs)."""
    out = json.loads(json.dumps(a))
    for tax, tags in b.get("facts", {}).items():
        dst = out.setdefault("facts", {}).setdefault(tax, {})
        for tag, body in tags.items():
            d = dst.setdefault(tag, {"units": {}})
            for unit, ents in body.get("units", {}).items():
                cur = d["units"].setdefault(unit, [])
                seen = {(e.get("start"), e["end"], e["filed"], e.get("form")) for e in cur}
                cur += [e for e in ents if (e.get("start"), e["end"], e["filed"], e.get("form")) not in seen]
    return out


def company_facts(ticker: str, refresh: bool = False) -> dict:
    path = _cache_path(f"facts_v2_{ticker}.json")
    if os.path.exists(path) and not refresh:
        with open(path) as f:
            return json.load(f)
    ciks = [_cik_map()[ticker.replace("-", ".").upper()]]
    ciks += [c for c in C.EXTRA_CIKS.get(ticker, []) if c not in ciks]
    data = None
    for cik in ciks:
        try:
            d = _sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
        except Exception as e:  # an extra CIK may have no XBRL facts
            print(f"  {ticker}: CIK {cik} unavailable ({e})")
            continue
        data = d if data is None else _merge_facts(data, d)
    with open(path, "w") as f:
        json.dump(data, f)
    return data


_FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT", "20-F", "40-F"}


def _entries(facts: dict, tag: str, taxonomy: str = "us-gaap", unit: str = "USD") -> list[dict]:
    try:
        return [e for e in facts["facts"][taxonomy][tag]["units"][unit] if e.get("form") in _FORMS]
    except KeyError:
        return []


def _quarters_for_tag(facts: dict, tag: str, unit: str = "USD", additive: bool = True) -> pd.DataFrame:
    """
    Quarterly values for a duration item, point-in-time.
    Returns columns: end, val, filed.
    additive=True  (income statement): Q4 is derived as FY minus Q1-Q3.
    additive=False (weighted-average share counts): Q4 can't be derived by
    subtraction, so the fiscal-year figure stands in for the Q4 period.
    """
    rows = {}
    annual = {}
    for e in _entries(facts, tag, unit=unit):
        if "start" not in e:
            continue
        s, en, f = pd.Timestamp(e["start"]), pd.Timestamp(e["end"]), pd.Timestamp(e["filed"])
        days = (en - s).days
        if 80 <= days <= 100:
            key = (s, en)
            if key not in rows or f < rows[key][1]:
                rows[key] = (float(e["val"]), f)
        elif 350 <= days <= 380:
            key = (s, en)
            if key not in annual or f < annual[key][1]:
                annual[key] = (float(e["val"]), f)

    q = pd.DataFrame(
        [(s, en, v, f) for (s, en), (v, f) in rows.items()],
        columns=["start", "end", "val", "filed"],
    )

    derived = []
    for (s, en), (v, f) in annual.items():
        inside = q[(q["start"] >= s - pd.Timedelta(days=7)) & (q["end"] <= en + pd.Timedelta(days=7))]
        has_q4 = ((inside["end"] - en).abs() <= pd.Timedelta(days=7)).any()
        if has_q4:
            continue
        if not additive:
            derived.append((s, en, v, f))
            continue
        if len(inside) == 3:
            # Q4 is only knowable once the 10-K is filed
            derived.append((inside["end"].max() + pd.Timedelta(days=1), en,
                            v - inside["val"].sum(), max(f, inside["filed"].max())))
    if derived:
        q = pd.concat([q, pd.DataFrame(derived, columns=q.columns)], ignore_index=True)

    if q.empty:
        return q
    q = q.sort_values(["end", "filed"]).drop_duplicates("end", keep="first")
    return q[["end", "val", "filed"]].reset_index(drop=True)


def _quarters(facts: dict, tags: list[str], unit: str = "USD", additive: bool = True) -> pd.DataFrame:
    """Union of several tags; earlier tags in the list take priority per period."""
    out = None
    for tag in tags:
        q = _quarters_for_tag(facts, tag, unit, additive)
        if q.empty:
            continue
        if out is None:
            out = q
        else:
            have = out["end"]
            near = q["end"].apply(lambda d: ((have - d).abs() <= pd.Timedelta(days=10)).any())
            out = pd.concat([out, q[~near]], ignore_index=True)
    if out is None:
        return pd.DataFrame(columns=["end", "val", "filed"])
    return out.sort_values("end").reset_index(drop=True)


def _instants(facts: dict, tags: list[str], taxonomy: str = "us-gaap", unit: str = "USD") -> pd.DataFrame:
    """Balance-sheet (point-in-time) items. Earlier tags take priority per date."""
    rows = {}
    for prio, tag in enumerate(tags):
        for e in _entries(facts, tag, taxonomy, unit):
            if "start" in e:
                continue
            en, f = pd.Timestamp(e["end"]), pd.Timestamp(e["filed"])
            cur = rows.get(en)
            cand = (prio, f, float(e["val"]))
            if cur is None or (cand[0], cand[1]) < (cur[0], cur[1]):
                rows[en] = cand
    if not rows:
        return pd.DataFrame(columns=["end", "val", "filed"])
    df = pd.DataFrame([(en, v, f) for en, (_, f, v) in rows.items()], columns=["end", "val", "filed"])
    return df.sort_values("end").reset_index(drop=True)


def _ttm_asof(q: pd.DataFrame, d: pd.Timestamp) -> float:
    if q.empty:
        return np.nan
    known = q[q["filed"] <= d]
    if len(known) < 4:
        return np.nan
    last4 = known.iloc[-4:]
    span = (last4["end"].iloc[-1] - last4["end"].iloc[0]).days
    stale = (d - last4["end"].iloc[-1]).days
    if span > 300 or span < 240 or stale > 200:
        return np.nan
    return float(last4["val"].sum())


def _latest_asof(df: pd.DataFrame, d: pd.Timestamp, max_age_days: int = 200):
    if df.empty:
        return np.nan, None
    known = df[df["filed"] <= d]
    if known.empty:
        return np.nan, None
    row = known.iloc[-1]
    if (d - row["end"]).days > max_age_days:
        return np.nan, None
    return float(row["val"]), row["end"]


TAGS = {
    "ebit": ["OperatingIncomeLoss"],
    "ni": ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss"],
    "shares_q": ["WeightedAverageNumberOfDilutedSharesOutstanding",
                 "WeightedAverageNumberOfSharesOutstandingBasic"],
    "ltd_total": ["LongTermDebt"],                       # includes current portion
    "ltd_nc": ["LongTermDebtNoncurrent"],
    "ltd_cur": ["LongTermDebtCurrent", "DebtCurrent"],
    "std": ["ShortTermBorrowings", "CommercialPaper"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash"],
}


def fundamentals_for(ticker: str, dates: pd.DatetimeIndex, prices: PriceData) -> pd.DataFrame:
    """Monthly point-in-time EBIT/EV and E/P for one ticker."""
    facts = company_facts(ticker)
    ebit_q = _quarters(facts, TAGS["ebit"])
    ni_q = _quarters(facts, TAGS["ni"])
    sh_q = _quarters(facts, TAGS["shares_q"], unit="shares", additive=False)
    sh_dei = _instants(facts, ["EntityCommonStockSharesOutstanding"], taxonomy="dei", unit="shares")
    sh_inst = _instants(facts, ["CommonStockSharesOutstanding"], unit="shares")
    floats = _instants(facts, ["EntityPublicFloat"], taxonomy="dei")
    ltd_total = _instants(facts, TAGS["ltd_total"])
    ltd_nc = _instants(facts, TAGS["ltd_nc"])
    ltd_cur = _instants(facts, TAGS["ltd_cur"])
    std = _instants(facts, TAGS["std"])
    cash = _instants(facts, TAGS["cash"])

    close = prices.close[ticker]
    out = []
    for d in dates:
        px = close.loc[:d].dropna()
        if px.empty:
            out.append((d, np.nan, np.nan, np.nan))
            continue
        p = float(px.iloc[-1])

        shares, asof = _latest_asof(sh_q, d)
        if np.isnan(shares):
            shares, asof = _latest_asof(sh_dei, d)
        if np.isnan(shares):
            shares, asof = _latest_asof(sh_inst, d)
        if not np.isnan(shares):
            mcap = shares * p * prices.splits_after(ticker, asof)
        else:
            # Last resort (Visa reports no total share count): scale the annual
            # public float by the price move since its measurement date.
            flt, fdate = _latest_asof(floats, d, max_age_days=600)
            pf = close.loc[:fdate].dropna() if fdate is not None else pd.Series(dtype=float)
            if np.isnan(flt) or pf.empty:
                out.append((d, np.nan, np.nan, np.nan))
                continue
            mcap = flt * p / float(pf.iloc[-1])

        ni = _ttm_asof(ni_q, d)
        ebit = _ttm_asof(ebit_q, d)

        # Total long-term debt if reported; otherwise non-current + current portion.
        debt_s, _ = _latest_asof(std, d)
        tot, _ = _latest_asof(ltd_total, d)
        if not np.isnan(tot):
            debt_lt, debt_c, has_debt = tot, 0.0, True
        else:
            debt_lt, _ = _latest_asof(ltd_nc, d)
            debt_c, _ = _latest_asof(ltd_cur, d)
            has_debt = not np.isnan(debt_lt)
            debt_lt = np.nan_to_num(debt_lt)
        c, _ = _latest_asof(cash, d)

        ev = np.nan
        if has_debt and not np.isnan(c):
            ev = mcap + debt_lt + np.nan_to_num(debt_c) + np.nan_to_num(debt_s) - c

        ep = ni / mcap if mcap > 0 else np.nan
        ev_ebit = ebit / ev if (ev and ev > 0 and not np.isnan(ebit)) else np.nan
        out.append((d, ep, ev_ebit, mcap))

    return pd.DataFrame(out, columns=["date", "ep", "ev_ebit", "mcap"]).set_index("date")


def load_fundamentals(prices: PriceData, dates: pd.DatetimeIndex, refresh: bool = False) -> dict:
    """
    Returns dict metric -> DataFrame (dates x tickers) of yields.
    Higher yield = cheaper.
    """
    path = _cache_path("fundamentals.pkl")
    if os.path.exists(path) and not refresh:
        with open(path, "rb") as f:
            cached = pickle.load(f)
        if cached["ep"].index.equals(dates):
            return cached

    per = {}
    for t in universe():
        print(f"  fundamentals: {t}")
        per[t] = fundamentals_for(t, dates, prices)
    res = {m: pd.DataFrame({t: per[t][m] for t in per}) for m in ("ep", "ev_ebit", "mcap")}
    with open(path, "wb") as f:
        pickle.dump(res, f)
    return res


# ---------------------------------------------------------------------------
# Synthetic data (for testing the pipeline offline)
# ---------------------------------------------------------------------------
def synthetic(seed: int = 7) -> tuple[PriceData, pd.DataFrame]:
    """
    Fake universe with the same tickers. Each pair shares a sector factor and a
    mean-reverting spread; three pairs 'break' partway through (the spread
    becomes a random walk) so the breakdown diagnostics have something to find.
    Yields are tied to the price spread (cheap price -> high yield) plus noise.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(C.DATA_START, "2026-09-30")
    n = len(idx)
    mkt = rng.normal(0.0004, 0.011, n)
    adj, yields = {}, {}
    breaks = {2: "2017-06-30", 9: "2019-01-31", 13: "2021-03-31"}
    for k, (a, b, _) in enumerate(C.PAIRS):
        sector = rng.normal(0, 0.007, n)
        ou = np.zeros(n)
        brk = idx.searchsorted(pd.Timestamp(breaks[k])) if k in breaks else n
        for i in range(1, n):
            kappa = 0.02 if i < brk else 0.0
            ou[i] = ou[i - 1] * (1 - kappa) + rng.normal(0, 0.006)
        beta = rng.uniform(0.7, 1.2)          # siblings share a beta, so the spread
        common = np.cumsum(beta * mkt + sector)  # carries no market random walk
        for sign, t in ((+0.5, a), (-0.5, b)):
            idio = np.zeros(n)
            eps = rng.normal(0, 0.006, n)
            for i in range(1, n):
                idio[i] = idio[i - 1] * 0.97 + eps[i]
            logp = common + idio + sign * ou
            adj[t] = 50 * np.exp(logp)
            base = rng.uniform(0.04, 0.08)
            yields[t] = base * np.exp(-sign * ou * 1.2 + rng.normal(0, 0.05, n))
    spy_r = mkt + rng.normal(0, 0.002, n)
    adj[C.BENCHMARK] = 100 * np.exp(np.cumsum(spy_r))
    adj = pd.DataFrame(adj, index=idx)
    y = pd.DataFrame(yields, index=idx).resample(C.REBALANCE).last()
    return PriceData(adj=adj, close=adj.copy(), splits={}), y
