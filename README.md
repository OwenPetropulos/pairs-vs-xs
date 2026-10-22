# Pairs vs. Cross-Sectional Relative Value

Does comparing a stock to its **sibling** (KO vs PEP, HD vs LOW) tell you more than comparing it to the **whole universe**? Both strategies here are built identically; the only difference is the reference each stock is measured against. Each runs on two signals: a **fundamental** one (EBIT/EV gap from point-in-time SEC data) and a **price-spread** one (log price gap).

## Run it

```bash
pip install -r requirements.txt
```

1. Open `pvx/config.py` and set `SEC_USER_AGENT` to your name and email (SEC requires it).
2. `python tests.py` — sanity checks (EDGAR parsing, no look-ahead, breakdown detection).
3. `python run.py` — downloads prices and fundamentals (first run takes a few minutes; everything is cached in `data_cache/`), runs the backtest, writes `results/report.html` plus CSVs.

`python run.py --synthetic` runs the whole pipeline on fake data, offline. `python run.py --refresh` re-downloads.

## What it compares

| Strategy | Reference | Book |
|---|---|---|
| Pairs L/S | sibling | long cheap / short rich within each pair when \|z\| ≥ 1, exit < 0.25, stop > 3 |
| Cross-sectional L/S | universe | long cheapest third / short richest third |
| Pairs long-only | sibling | hold the cheap sibling of every pair |
| Cross-sectional long-only | universe | hold the cheapest third |
| Equal weight | — | all 30 names, monthly |
| SPY | — | buy and hold |

Signals are set at month-end close and traded at the next day's close. 5bp trading cost, 30bp/yr borrow on shorts.

## Breakdown diagnostics

For each pair, in `results/pair_summary_*.csv` and the report:

- **Contribution, 1st vs 2nd half** — did the pair stop paying?
- **Half-life, 1st vs 2nd half** — how fast the gap closes (AR(1) on the monthly gap). A half-life that lengthens, or shows `none`, means the pair stopped mean-reverting.
- **Return correlation, 1st vs 2nd half** — did the siblings stop moving together?
- **Stop-outs and their P&L** — trades where the gap kept widening past 3σ.
- **Converged ≤6m** — share of trades where the gap actually closed.

The pair explorer in the report plots each pair's z-scores, entry/stop bands, rolling correlation and P&L together, which is the quickest way to find the story behind a pair that broke.

## Caveats to state in the writeup

- The 15 pairs are chosen today, so they all survived. That flatters the results.
- XBRL tags vary across companies and years. `run.py` prints EBIT/EV coverage per ticker; pairs below 80% fall back to E/P.
- Market cap = diluted weighted-average shares × price, adjusted for later splits.
- Cash earns nothing, so Sharpe ratios are on total returns.
