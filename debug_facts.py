"""
Writes a compact copy of the cached SEC companyfacts (only tags relevant to
valuation, only 10-K/10-Q filings, only the fields the parser uses) so the
fundamental pipeline can be debugged offline.

    python debug_facts.py   ->  results/debug/facts_compact.json.gz
"""
import glob
import gzip
import json
import os
import re

KEEP = re.compile(r"(OperatingIncome|NetIncome|ProfitLoss|SharesOutstanding|Debt|Borrowing|"
                  r"CommercialPaper|Cash|Revenue|IncomeLossFromContinuing|DepreciationDepletion|EarningsPerShare)", re.I)
FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT"}

out = {}
for path in sorted(glob.glob("data_cache/facts_*.json")):
    t = os.path.basename(path)[:-5].split("_")[-1]
    with open(path) as f:
        facts = json.load(f)["facts"]
    keep = {}
    for tax, tags in facts.items():
        if tax not in ("us-gaap", "dei"):
            continue
        for tag, body in tags.items():
            if tax == "us-gaap" and not KEEP.search(tag):
                continue
            units = {}
            for u, ents in body.get("units", {}).items():
                rows = [[e.get("start"), e["end"], e["val"], e["filed"], e.get("form")]
                        for e in ents if e.get("form") in FORMS]
                if rows:
                    units[u] = rows
            if units:
                keep[f"{tax}:{tag}"] = units
    out[t] = keep

os.makedirs("results/debug", exist_ok=True)
with gzip.open("results/debug/facts_compact.json.gz", "wt") as f:
    json.dump(out, f)
print({t: len(v) for t, v in out.items()})
