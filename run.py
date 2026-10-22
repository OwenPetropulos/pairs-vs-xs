"""
Pairs vs. Cross-Sectional relative value backtest.

    python run.py              # real data (yfinance + SEC EDGAR), cached after first run
    python run.py --refresh    # re-download everything
    python run.py --synthetic  # offline pipeline test on fake data

Outputs go to results/: report.html plus CSVs of every table.
"""
import argparse
import os
import sys

import pandas as pd

from pvx import backtest, config as C, data, diagnostics as D, report, signals


def main():
    try:  # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    tickers = data.universe()
    print("1/5 prices")
    if args.synthetic:
        prices, syn_y = data.synthetic()
        yields = {"ev_ebit": syn_y, "ep": syn_y * 0.8}
    else:
        prices = data.load_prices(refresh=args.refresh)
        print("2/5 fundamentals (SEC EDGAR, point-in-time)")
        fdates = signals.rebalance_dates(prices.adj.index)
        yields = data.load_fundamentals(prices, fdates, refresh=args.refresh)
        cov = yields["ep"].loc[C.BACKTEST_START:].notna().mean()
        print("  E/P coverage since backtest start:")
        print(cov.sort_values().to_string(float_format=lambda x: f"{x:.0%}"))

    print("3/5 signals")
    sig = signals.build_all(prices, yields, tickers)

    print("4/5 backtest")
    bt = backtest.run_all(prices, sig, tickers)
    res = bt["results"]

    print("5/5 diagnostics + report")
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    M = D.metrics(res)
    sub = D.subperiods(res)
    roll = D.rolling_sharpe(res, ["fund_pairs_ls", "fund_xs_ls", "price_pairs_ls", "price_xs_ls"])
    corr = D.strategy_correlations(res)
    pair_corr = D.rolling_pair_corr(prices.adj)
    S = sig["signals"]

    attrs, tables, summaries = {}, {}, {}
    for src in ("fund", "price"):
        key = f"{src}_pairs_ls"
        attrs[src] = D.pair_attribution(res, key)
        tables[src] = D.trade_table(bt["trades"][src], res, key, S[f"{src}_pairs"])
        summaries[src] = D.pair_summary(res, tables[src], key, S[f"{src}_pairs"], pair_corr,
                                        sig["gaps"][src])

    found = D.findings(M, summaries, tables, sub)

    out = C.OUTPUT_DIR
    M.to_csv(f"{out}/metrics.csv")
    sub.to_csv(f"{out}/subperiod_sharpe.csv")
    corr.to_csv(f"{out}/strategy_correlations.csv")
    pd.DataFrame({k: r["nav"] for k, r in res.items()}).to_csv(f"{out}/nav.csv")
    for src in ("fund", "price"):
        summaries[src].to_csv(f"{out}/pair_summary_{src}.csv")
        tables[src].to_csv(f"{out}/trades_{src}.csv", index=False)
        S[f"{src}_pairs"].to_csv(f"{out}/signal_{src}_pairs.csv")
        S[f"{src}_xs"].to_csv(f"{out}/signal_{src}_xs.csv")

    report.build(f"{out}/report.html", res, M, sub, roll, corr, attrs, summaries, tables,
                 S, pair_corr, found, sig["metrics"], args.synthetic)

    pd.set_option("display.width", 200)
    print()
    print(M[["Strategy", "CAGR", "Vol", "Sharpe", "Max DD", "Beta (SPY)"]].to_string(
        float_format=lambda x: f"{x:.3f}"))
    print()
    for f in found:
        print(" -", f)
    print(f"\nReport: {os.path.abspath(out)}/report.html")


if __name__ == "__main__":
    main()
