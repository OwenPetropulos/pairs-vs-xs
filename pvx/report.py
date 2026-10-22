"""
Self-contained HTML report (Plotly from CDN), styled to the OPResearch system:
Fraunces headings, IBM Plex Mono for data, amber accent.
"""
from __future__ import annotations

import html
import json
import math

import numpy as np
import pandas as pd

from . import config as C
from .diagnostics import LABELS

AMBER = "#d98e04"
PAL = {
    "fund_pairs_ls": AMBER, "fund_xs_ls": "#7a5a12",
    "price_pairs_ls": "#2f6f8f", "price_xs_ls": "#9cc3d5",
    "fund_pairs_lo": AMBER, "fund_xs_lo": "#7a5a12",
    "price_pairs_lo": "#2f6f8f", "price_xs_lo": "#9cc3d5",
    "ew": "#8a8a8a", "spy": "#222222",
}
DASH = {k: ("dot" if "_xs_" in k else "solid") for k in PAL}
DASH.update(ew="dash", spy="solid")


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _xy(s: pd.Series):
    s = s.dropna()
    if len(s) > 1000:  # daily series -> weekly, keeps the page light
        s = s.resample("W-FRI").last().dropna()
    return [d.strftime("%Y-%m-%d") for d in s.index], [_clean(v) for v in s.values]


def _line(s, name, color, dash="solid", width=1.8, yaxis=None, visible=True, legend=True):
    x, y = _xy(s)
    t = {"type": "scatter", "mode": "lines", "x": x, "y": y, "name": name,
         "line": {"color": color, "dash": dash, "width": width}, "visible": visible,
         "showlegend": legend, "hovertemplate": "%{y:.3f}<extra>" + name + "</extra>"}
    if yaxis:
        t["yaxis"] = yaxis
    return t


def _layout(title=None, ylog=False, ytitle=None, height=420, **kw):
    lay = {
        "height": height,
        "margin": {"l": 56, "r": 20, "t": 30 if title else 10, "b": 40},
        "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "IBM Plex Mono, monospace", "size": 11, "color": "#2a2620"},
        "xaxis": {"gridcolor": "#e8e2d6", "linecolor": "#cfc6b5"},
        "yaxis": {"gridcolor": "#e8e2d6", "linecolor": "#cfc6b5", "zerolinecolor": "#cfc6b5",
                  "type": "log" if ylog else "linear", "title": {"text": ytitle or ""}},
        "legend": {"orientation": "h", "y": -0.15, "font": {"size": 10}},
        "hovermode": "x unified",
    }
    if title:
        lay["title"] = {"text": title, "font": {"family": "Fraunces, serif", "size": 15}}
    lay.update(kw)
    return lay


def _table(df: pd.DataFrame, fmt: dict, index_name="", highlight_col=None) -> str:
    cols = list(df.columns)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [index_name] + cols)
    body = []
    for idx, row in df.iterrows():
        cells = [f"<td class='lab'>{html.escape(str(idx))}</td>"]
        for c in cols:
            v = row[c]
            f = fmt.get(c, "{:.2f}")
            if v is None or (isinstance(v, float) and np.isnan(v)):
                s = "—"
                cls = ""
            elif isinstance(v, float) and np.isinf(v):
                s = "none"
                cls = "neg"
            else:
                s = f.format(v)
                cls = "neg" if isinstance(v, (int, float, np.number)) and v < 0 and c in fmt.get("_signed", []) else ""
            cells.append(f"<td class='{cls}'>{s}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='tbl'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def build(out_path: str, res: dict, M: pd.DataFrame, sub: pd.DataFrame, roll: pd.DataFrame,
          corr: pd.DataFrame, attrs: dict, summaries: dict, tables: dict, sigs: dict,
          pair_corr: pd.DataFrame, findings: list[str], metrics_used: dict, synthetic: bool,
          ic: pd.DataFrame | None = None):
    charts = {}

    # 1. Equity curves
    ls_keys = ["fund_pairs_ls", "fund_xs_ls", "price_pairs_ls", "price_xs_ls"]
    lo_keys = ["fund_pairs_lo", "fund_xs_lo", "price_pairs_lo", "price_xs_lo", "ew", "spy"]
    charts["eq_ls"] = {
        "data": [_line(res[k]["nav"], LABELS[k], PAL[k], DASH[k]) for k in ls_keys]
        + [_line(res["spy"]["nav"], LABELS["spy"], "#bbbbbb", "solid", 1.2)],
        "layout": _layout(ylog=True, ytitle="Growth of $1 (log)"),
    }
    charts["eq_lo"] = {
        "data": [_line(res[k]["nav"], LABELS[k], PAL[k], DASH[k], 2.4 if k in ("ew", "spy") else 1.8)
                 for k in lo_keys],
        "layout": _layout(ylog=True, ytitle="Growth of $1 (log)"),
    }

    # 2. Drawdowns
    charts["dd"] = {
        "data": [_line(res[k]["nav"] / res[k]["nav"].cummax() - 1, LABELS[k], PAL[k], DASH[k]) for k in ls_keys],
        "layout": _layout(ytitle="Drawdown", height=320, yaxis={"tickformat": ".0%", "gridcolor": "#e8e2d6"}),
    }

    # 3. Rolling Sharpe
    charts["roll"] = {
        "data": [_line(roll[k], LABELS[k], PAL[k], DASH[k]) for k in roll.columns],
        "layout": _layout(ytitle=f"Rolling {C.ROLLING_MONTHS}m Sharpe", height=340),
    }

    # 4. Pair contribution bars
    pairs = [f"{a}/{b}" for a, b, _ in C.PAIRS]
    charts["contrib"] = {
        "data": [
            {"type": "bar", "orientation": "h", "y": pairs,
             "x": [_clean(attrs["fund"][p].iloc[-1]) for p in pairs],
             "name": "Fundamental", "marker": {"color": AMBER}},
            {"type": "bar", "orientation": "h", "y": pairs,
             "x": [_clean(attrs["price"][p].iloc[-1]) for p in pairs],
             "name": "Spread", "marker": {"color": "#2f6f8f"}},
        ],
        "layout": _layout(height=520, barmode="group",
                          xaxis={"tickformat": ".0%", "gridcolor": "#e8e2d6", "title": {"text": "Cumulative contribution to Pairs L/S"}},
                          yaxis={"autorange": "reversed"}, hovermode="closest"),
    }

    # 5. Pair explorer: z-scores + rolling correlation, dropdown per pair
    data, buttons = [], []
    per = 4
    for i, p in enumerate(pairs):
        vis = i == 0
        data += [
            _line(sigs["fund_pairs"][p], "Fundamental z", AMBER, visible=vis),
            _line(sigs["price_pairs"][p], "Spread z", "#2f6f8f", visible=vis),
            _line(pair_corr[p], "1y return correlation", "#8a8a8a", "dot", 1.4, yaxis="y2", visible=vis),
            _line(attrs["fund"][p] * 100, "Fund. P&L contrib (%)", "#b04a2f", "solid", 1.2, yaxis="y3", visible=vis),
        ]
    for i, p in enumerate(pairs):
        v = [False] * (per * len(pairs))
        v[i * per:(i + 1) * per] = [True] * per
        buttons.append({"label": p, "method": "update", "args": [{"visible": v}]})
    lay = _layout(height=460, ytitle="Cheapness z (+ = first name cheap)")
    lay["yaxis"]["domain"] = [0.32, 1]
    lay["yaxis2"] = {"overlaying": "y", "side": "right", "range": [-0.2, 1], "showgrid": False,
                     "title": {"text": "Corr"}}
    lay["yaxis3"] = {"domain": [0, 0.24], "gridcolor": "#e8e2d6", "title": {"text": "P&L %"}}
    lay["shapes"] = [
        {"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": y, "y1": y, "yref": "y",
         "line": {"color": c, "width": 1, "dash": "dash"}}
        for y, c in ((C.ENTRY_Z, AMBER), (-C.ENTRY_Z, AMBER), (C.STOP_Z, "#b04a2f"), (-C.STOP_Z, "#b04a2f"))
    ]
    lay["updatemenus"] = [{"buttons": buttons, "x": 0, "y": 1.14, "xanchor": "left",
                           "bgcolor": "#fffaf0", "bordercolor": "#cfc6b5"}]
    lay["margin"]["t"] = 50
    charts["explorer"] = {"data": data, "layout": lay}

    # Tables
    mfmt = {"CAGR": "{:.1%}", "Vol": "{:.1%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
            "Max DD": "{:.1%}", "Calmar": "{:.2f}", "Beta (SPY)": "{:.2f}", "Corr (SPY)": "{:.2f}",
            "Avg gross": "{:.2f}", "Turnover/yr": "{:.1f}x", "% months up": "{:.0%}",
            "_signed": ["CAGR", "Sharpe", "Max DD"]}
    Mt = M.set_index("Strategy")
    t_metrics = _table(Mt, mfmt, "Strategy")
    t_sub = _table(sub, {c: "{:.2f}" for c in sub.columns} | {"_signed": list(sub.columns)}, "Sharpe by period")
    t_corr = _table(corr, {c: "{:.2f}" for c in corr.columns}, "Monthly return correlation")
    pfmt = {"Total contrib": "{:+.1%}", "Trades": "{:.0f}", "Hit rate": "{:.0%}", "Converged ≤6m": "{:.0%}",
            "Stop-outs": "{:.0f}", "Stop-out P&L": "{:+.1%}",
            "Contrib 1st half": "{:+.1%}", "Contrib 2nd half": "{:+.1%}",
            "Half-life 1st (m)": "{:.1f}", "Half-life 2nd (m)": "{:.1f}", "Avg hold (m)": "{:.1f}",
            "% months |z|>stop": "{:.0%}", "Corr 1st half": "{:.2f}", "Corr 2nd half": "{:.2f}",
            "_signed": ["Total contrib", "Stop-out P&L", "Contrib 1st half", "Contrib 2nd half"]}
    t_ic = ""
    if ic is not None:
        icd = ic.copy()
        icd.index = icd["Signal"] + " · " + icd["Horizon (m)"].astype(str) + "m"
        icd = icd[["Mean IC", "t-stat", "% months IC>0", "N"]]
        t_ic = _table(icd, {"Mean IC": "{:+.3f}", "t-stat": "{:+.2f}", "% months IC>0": "{:.0%}", "N": "{:.0f}",
                            "_signed": ["Mean IC", "t-stat"]}, "Signal · horizon")
    t_pf = _table(summaries["fund"], pfmt, "Pair")
    t_pp = _table(summaries["price"], pfmt, "Pair")

    fb = [p for p, m in metrics_used.items() if m == "ep"]
    metric_note = ", ".join(f"{a}/{b}" for a, b in fb) or "none"

    period = f"{res['spy']['nav'].index[0]:%b %Y} – {res['spy']['nav'].index[-1]:%b %Y}"
    banner = ("<div class='banner'>SYNTHETIC DATA — pipeline test only. Run <code>python run.py</code> "
              "without <code>--synthetic</code> for real results.</div>") if synthetic else ""
    find_html = "".join(f"<li>{html.escape(f)}</li>" for f in findings)

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pairs vs Cross-Section</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root {{ --bg:#faf7f0; --ink:#2a2620; --mute:#6b6458; --rule:#e2dacb; --amber:{AMBER}; --neg:#b04a2f; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:'IBM Plex Sans',system-ui,sans-serif; line-height:1.6; }}
main {{ max-width:1080px; margin:0 auto; padding:48px 16px 80px; }}
h1,h2,h3 {{ font-family:'Fraunces',serif; font-weight:600; letter-spacing:-0.01em; }}
h1 {{ font-size:2.4rem; margin:0 0 6px; line-height:1.15; }}
h2 {{ font-size:1.5rem; margin:56px 0 8px; padding-top:16px; border-top:1px solid var(--rule); }}
h3 {{ font-size:1.1rem; margin:28px 0 6px; }}
.kicker {{ font-family:'IBM Plex Mono',monospace; color:var(--amber); font-size:.8rem; text-transform:uppercase; letter-spacing:.12em; }}
.lede {{ color:var(--mute); max-width:720px; }}
.meta {{ font-family:'IBM Plex Mono',monospace; font-size:.78rem; color:var(--mute); }}
.chart {{ width:100%; }}
.tbl {{ overflow-x:auto; margin:12px 0; }}
table {{ border-collapse:collapse; font-family:'IBM Plex Mono',monospace; font-size:.76rem; width:100%; }}
th {{ text-align:right; font-weight:500; color:var(--mute); border-bottom:1px solid var(--ink); padding:6px 8px; white-space:nowrap; }}
th:first-child, td.lab {{ text-align:left; }}
td {{ text-align:right; padding:5px 8px; border-bottom:1px solid var(--rule); white-space:nowrap; }}
td.neg {{ color:var(--neg); }}
ul.find li {{ margin:6px 0; }}
.banner {{ background:#fff1d6; border:1px solid var(--amber); padding:10px 14px; font-family:'IBM Plex Mono',monospace; font-size:.8rem; margin:16px 0; }}
code {{ font-family:'IBM Plex Mono',monospace; font-size:.85em; }}
.note {{ color:var(--mute); font-size:.9rem; max-width:760px; }}
</style></head>
<body><main>
<div class="kicker">OPResearch · Relative value</div>
<h1>Sibling or Universe? Pairs Trading vs. Cross-Sectional Value</h1>
<p class="lede">Each stock is scored on how cheap it is relative to a reference. The pairs strategy uses its
sibling (KO vs PEP, HD vs LOW); the cross-sectional strategy uses the whole universe. Everything else is
held constant, so the gap between them measures what the sibling comparison is worth.</p>
<div class="meta">{period} · {len(C.PAIRS)} pairs · monthly rebalance · {C.TC_BPS:.0f}bp trading cost · {C.BORROW_BPS:.0f}bp/yr borrow</div>
{banner}

<h2>Findings</h2>
<ul class="find">{find_html}</ul>

<h2>Long/short: pairs vs. cross-section</h2>
<p class="note">Solid = pairs, dotted = cross-sectional. Amber = fundamental signal, blue = price-spread signal.
These books are roughly market-neutral, so judge them on Sharpe and drawdown, not on whether they beat SPY.</p>
<div id="eq_ls" class="chart"></div>
<h3>Drawdowns</h3>
<div id="dd" class="chart"></div>

<h2>Long-only vs. benchmarks</h2>
<p class="note">Holding only the cheap side lets each approach be compared fairly with the equal-weight universe and SPY.</p>
<div id="eq_lo" class="chart"></div>

<h2>Performance</h2>
{t_metrics}
{t_sub}
{t_corr}

<h2>Does cheapness predict anything?</h2>
<p class="note">Before any trading rule: the rank correlation between each month's cheapness score and the
forward relative return. Positive means cheap names went on to outperform. Windows don't overlap.</p>
{t_ic}

<h2>Is the edge stable?</h2>
<div id="roll" class="chart"></div>

<h2>Where it broke down</h2>
<p class="note">Cumulative contribution of each pair to the Pairs L/S book. Pairs with negative bars are the
starting points for the breakdown analysis: open them in the explorer below and look for a z-score that
trends instead of mean-reverting, or a return correlation that falls.</p>
<div id="contrib" class="chart"></div>
<h3>Pair explorer</h3>
<p class="note">Dashed amber = entry band (±{C.ENTRY_Z}), dashed red = stop (±{C.STOP_Z}). Grey dotted = rolling 1-year
return correlation (right axis). Bottom panel = this pair's fundamental P&amp;L contribution.</p>
<div id="explorer" class="chart"></div>
<h3>Pair diagnostics · fundamental signal</h3>
{t_pf}
<h3>Pair diagnostics · spread signal</h3>
{t_pp}

<h2>Method</h2>
<div class="note">
<p><b>Signals.</b> Fundamental: earnings yield (TTM net income / market cap) from SEC EDGAR, rebuilt
point-in-time by filing date. EBIT/EV was the original plan, but oil majors, pharma and banks report no
operating-income line and debt tags change names over time, so E/P is the one measure every name supports. Pairs gap = yield<sub>A</sub> − yield<sub>B</sub>; cross-sectional
gap = yield<sub>i</sub> − universe median. Each gap is z-scored over a trailing {C.FUND_Z_WINDOW} months.
Spread: pairs gap = log P<sub>A</sub> − log P<sub>B</sub>; cross-sectional gap = log P<sub>i</sub> − mean log P;
z-scored over {C.PRICE_Z_WINDOW} trading days and sign-flipped so positive = cheap.</p>
<p><b>Portfolios.</b> Pairs L/S: each pair has a fixed 1/{len(C.PAIRS)} slot; enter at |z| ≥ {C.ENTRY_Z}, exit at
|z| &lt; {C.EXIT_Z} or a sign flip, stop at |z| &gt; {C.STOP_Z} and stay out until the gap resets. Cross-sectional
L/S: long cheapest third, short richest third, 100% of capital per leg. Signals at month-end close, trades at the
next close. Cash earns zero, so Sharpe ratios are on total, not excess, returns.</p>
<p><b>Known biases.</b> The pair list is chosen with hindsight (all 15 pairs survive to today). EDGAR XBRL tags vary by
company and year, so some fundamental gaps are noisy. Market cap uses diluted weighted shares from the latest quarter;
Visa reports no total share count, so its market cap is scaled from its annual public float, which leaves out
its class B and C shares.</p>
</div>
</main>
<script>
const charts = {json.dumps(charts)};
const cfg = {{responsive:true, displaylogo:false, modeBarButtonsToRemove:['lasso2d','select2d']}};
for (const [id, c] of Object.entries(charts)) Plotly.newPlot(id, c.data, c.layout, cfg);
</script>
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
