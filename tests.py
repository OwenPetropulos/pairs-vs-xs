"""
Sanity checks. Run:  python tests.py

1. EDGAR parser: point-in-time TTM, derived Q4, restatements ignored.
2. Engine: a signal that peeks one month ahead must look far better than the
   honest one (proves the lag is real), and random signals must earn ~nothing.
3. Synthetic pairs that 'break' should show a longer spread half-life afterwards.
"""
import numpy as np
import pandas as pd

from pvx import backtest, config as C, data, diagnostics as D, signals


def fake_facts():
    def q(s, e, v, f, form="10-Q"):
        return {"start": s, "end": e, "val": v, "filed": f, "form": form}
    ebit = [
        q("2020-01-01", "2020-03-31", 10, "2020-05-01"),
        q("2020-04-01", "2020-06-30", 11, "2020-08-01"),
        q("2020-07-01", "2020-09-30", 12, "2020-11-01"),
        q("2020-01-01", "2020-12-31", 50, "2021-02-20", "10-K"),       # Q4 derived = 17
        q("2021-01-01", "2021-03-31", 13, "2021-05-01"),
        q("2021-01-01", "2021-03-31", 99, "2022-05-01", "10-Q/A"),     # later restatement: ignore
        q("2020-01-01", "2020-06-30", 21, "2020-08-01"),               # YTD 6m: ignore
    ]
    return {"facts": {"us-gaap": {"OperatingIncomeLoss": {"units": {"USD": ebit}}}}}


def test_edgar():
    f = fake_facts()
    qd = data._quarters(f, ["OperatingIncomeLoss"])
    assert list(qd["val"]) == [10, 11, 12, 17, 13], qd
    assert np.isnan(data._ttm_asof(qd, pd.Timestamp("2021-01-31")))         # Q4 not yet filed
    assert data._ttm_asof(qd, pd.Timestamp("2021-03-01")) == 50              # Q1-Q4 2020
    assert data._ttm_asof(qd, pd.Timestamp("2021-06-01")) == 11 + 12 + 17 + 13
    sh = {"facts": {"us-gaap": {"WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": [
        {"start": "2020-01-01", "end": "2020-03-31", "val": 100, "filed": "2020-05-01", "form": "10-Q"},
        {"start": "2020-04-01", "end": "2020-06-30", "val": 101, "filed": "2020-08-01", "form": "10-Q"},
        {"start": "2020-07-01", "end": "2020-09-30", "val": 102, "filed": "2020-11-01", "form": "10-Q"},
        {"start": "2020-01-01", "end": "2020-12-31", "val": 101.5, "filed": "2021-02-20", "form": "10-K"},
    ]}}}}}
    sq = data._quarters(sh, ["WeightedAverageNumberOfDilutedSharesOutstanding"], unit="shares", additive=False)
    v, asof = data._latest_asof(sq, pd.Timestamp("2021-03-01"))
    assert v == 101.5 and asof == pd.Timestamp("2020-12-31"), (v, asof)
    print("edgar parser: ok")


def test_engine():
    prices, y = data.synthetic(seed=11)
    tickers = data.universe()
    dates = signals.rebalance_dates(prices.adj.index)
    dates = dates[dates >= pd.Timestamp(C.BACKTEST_START)]
    # next-month realised return of each stock = perfect foresight
    m = prices.adj[tickers].reindex(dates)
    fwd = m.shift(-1) / m - 1                           # next month return, roughly what is earned
    peek = backtest.simulate(backtest.xs_ls(fwd.iloc[:-1], tickers), prices.adj)
    rng = np.random.default_rng(0)
    rand = pd.DataFrame(rng.normal(size=fwd.shape), index=fwd.index, columns=tickers).iloc[:-1]
    randr = backtest.simulate(backtest.xs_ls(rand, tickers), prices.adj)

    def sharpe(nav):
        d = nav.pct_change().dropna()
        return d.mean() / d.std() * np.sqrt(252)

    sp, sr = sharpe(peek["nav"]), sharpe(randr["nav"])
    print(f"engine: peeking Sharpe {sp:.2f}, random Sharpe {sr:.2f}")
    assert sp > 3, "a look-ahead signal should be spectacular; the lag may be broken"
    assert abs(sr) < 0.8, "random signals should earn roughly nothing"


def test_breaks():
    prices, _ = data.synthetic()
    logp = np.log(prices.adj).resample("ME").last()
    for k in (2, 9, 13):
        a, b, _ = C.PAIRS[k]
        g = logp[a] - logp[b]
        h1 = D.half_life(g.loc[:"2016-12-31"])
        brk = {2: "2017-07-01", 9: "2019-02-01", 13: "2021-04-01"}[k]
        h2 = D.half_life(g.loc[brk:])
        print(f"  {a}/{b}: half-life before {h1:.1f}m, after break {h2:.1f}m")
        assert h2 > 2 * h1
    print("breakdown detection: ok")


if __name__ == "__main__":
    test_edgar()
    test_engine()
    test_breaks()
