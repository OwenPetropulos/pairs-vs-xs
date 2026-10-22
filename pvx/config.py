"""
Configuration for the Pairs vs. Cross-Sectional relative-value backtest.

Edit this file, not the code. Everything the backtest depends on lives here.
"""

# ---------------------------------------------------------------------------
# SEC EDGAR requires a descriptive User-Agent with a contact email.
# Fill in your own before running with real data, e.g.
#   "Owen Petropulos owen@example.com"
# ---------------------------------------------------------------------------
import os
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "YOUR NAME your_email@example.com")

# ---------------------------------------------------------------------------
# Universe: sibling pairs, fixed ex ante.
#
# metric:
#   "ev_ebit" -> operating earnings yield on enterprise value, EBIT / EV
#   "ep"      -> earnings yield, TTM net income / market cap
# Banks use "ep" because EV and operating income are not meaningful for them.
#
# Selection-bias warning: these pairs are chosen today. Every one of them still
# exists in 2026 as a recognisable pair, which flatters the backtest. Say so in
# the writeup.
# ---------------------------------------------------------------------------
PAIRS = [
    ("KO",  "PEP", "ev_ebit"),   # beverages
    ("HD",  "LOW", "ev_ebit"),   # home improvement
    ("MCD", "YUM", "ev_ebit"),   # QSR (Burger King sits inside QSR/RBI; YUM is cleaner)
    ("V",   "MA",  "ev_ebit"),   # card networks
    ("UPS", "FDX", "ev_ebit"),   # parcel
    ("XOM", "CVX", "ev_ebit"),   # integrated oil
    ("JPM", "BAC", "ep"),        # money-centre banks
    ("CAT", "DE",  "ev_ebit"),   # heavy machinery
    ("LMT", "NOC", "ev_ebit"),   # defense primes
    ("DAL", "UAL", "ev_ebit"),   # airlines
    ("UNP", "CSX", "ev_ebit"),   # rail
    ("WMT", "TGT", "ev_ebit"),   # big-box retail
    ("PG",  "CL",  "ev_ebit"),   # household products
    ("MRK", "PFE", "ev_ebit"),   # pharma
    ("T",   "VZ",  "ev_ebit"),   # telecom
]

BENCHMARK = "SPY"

# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
DATA_START = "2008-01-01"      # history used to warm up signals
BACKTEST_START = "2012-01-01"  # first rebalance (XBRL fundamentals need warm-up)
BACKTEST_END = None            # None = latest available

# ---------------------------------------------------------------------------
# Signal parameters
# ---------------------------------------------------------------------------
REBALANCE = "ME"               # month-end rebalancing

# Fundamental signal: z-score of the relative-yield gap vs its own trailing history
FUND_Z_WINDOW = 36             # months of history in the z-score
FUND_Z_MIN = 24                # minimum months before a signal is produced

# Spread (price) signal: z-score of the log price gap vs its trailing history
PRICE_Z_WINDOW = 252           # trading days

# Pairs long/short trading rules (applied to |z|)
ENTRY_Z = 1.0                  # open a position when the gap is this stretched
EXIT_Z = 0.25                  # close once the gap has mostly closed
STOP_Z = 3.0                   # treat beyond this as "broken"; go flat
EXEC_LAG_DAYS = 1              # signal at month-end close, trade next day's close

# Pairs whose ev_ebit coverage is below this fall back to E/P for both legs
MIN_COVERAGE = 0.8

# Cross-sectional: fraction of the universe in each leg
XS_QUANTILE = 1 / 3

# ---------------------------------------------------------------------------
# Frictions
# ---------------------------------------------------------------------------
TC_BPS = 5.0                   # one-way trading cost per $ traded, in bps
BORROW_BPS = 30.0              # annual borrow cost on short notional, in bps

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
SUBPERIODS = [
    ("2012-01-01", "2015-12-31"),
    ("2016-01-01", "2019-12-31"),
    ("2020-01-01", "2022-12-31"),
    ("2023-01-01", "2030-12-31"),
]
CONVERGE_Z = 0.5               # a trade "converged" if |z| falls below this
CONVERGE_HORIZON = 6           # ...within this many months
ROLLING_MONTHS = 36            # rolling Sharpe window

CACHE_DIR = "data_cache"
OUTPUT_DIR = "results"
