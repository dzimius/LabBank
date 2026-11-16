"""Download selected PDMA daily market reports and extract the EUR IRS row.

usage: python -I pdma_extract.py <index_csv> <download_dir> <out_csv> [--test d1,d2,...]

Selection: every report from 2026-01-01, plus the reports closest to the 15th and the
last business day of each month before that. Output: one row per report date with the
EUR IRS tenors found in that report (percent).
"""
import csv, io, re, sys, time
from pathlib import Path

import truststore; truststore.inject_into_ssl()
import pandas as pd
import requests
from pypdf import PdfReader

idx_csv, dl_dir, out_csv = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
test = sys.argv[sys.argv.index("--test") + 1].split(",") if "--test" in sys.argv else None
dl_dir.mkdir(parents=True, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0 (LabBank research)"}
BASE = "https://www.pdma.gr"

idx = pd.read_csv(idx_csv, parse_dates=["date"]).drop_duplicates("date", keep="last").sort_values("date")
if test:
    sel = idx[idx.date.isin(pd.to_datetime(test))]
else:
    recent = idx[idx.date >= "2026-01-01"]
    old = idx[idx.date < "2026-01-01"].copy()
    old["ym"] = old.date.dt.to_period("M")
    picks = []
    for _, g in old.groupby("ym"):
        mid = g.iloc[(g.date.dt.day - 15).abs().argsort()[:1]]
        picks += [mid, g.tail(1)]
    sel = pd.concat([pd.concat(picks), recent]).drop_duplicates("date").sort_values("date")
print("reports selected:", len(sel), sel.date.min().date(), "..", sel.date.max().date(), flush=True)

TENOR = re.compile(r"(\d+)\s*y(?:\s*e\s*a\s*r\s*s?)?\b", re.I)   # "1 year", "30 YEA R", "10Y"
NUM = re.compile(r"-?\d+[,.]\d+")


def extract(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines()]
    for i, l in enumerate(lines):
        if re.match(r"EUR\s*(?:IRS|SWAP)\b", l, re.I):   # 2024+: "EUR IRS", before: "EUR SWAP"
            vals = [float(v.replace(",", ".")) for v in NUM.findall(l)]
            for j in range(i - 1, max(-1, i - 6), -1):        # nearest tenor header above
                ten = [int(m.group(1)) for m in TENOR.finditer(lines[j])]
                if len(ten) >= 3:
                    break
            else:
                continue
            if len(vals) == len(ten):
                return {f"{t}Y": v for t, v in zip(ten, vals)}
    return {}


rows, fails = [], []
for k, r in enumerate(sel.itertuples()):
    try:
        page = requests.get(r.page_url, headers=H, timeout=60).text
        links = (re.findall(r'href="([^"]+\.pdf)"', page, re.I)          # 2026 layout
                 + re.findall(r"href='([^']+\.pdf)'", page, re.I)         # 2018-2020 attachments
                 + re.findall(r'href="([^"]*phocadownload[^"]*download=\d+[^"]*)"', page))  # 2020-2025
        if not links:
            fails.append((r.date.date(), "no pdf link")); continue
        url = links[0].replace("&amp;", "&")
        url = url if url.startswith("http") else BASE + url
        f = dl_dir / f"pdma_{r.date.date()}.pdf"
        if not f.exists():
            resp = requests.get(url, headers=H, timeout=120)
            if not resp.content.startswith(b"%PDF"):
                fails.append((r.date.date(), f"not a pdf ({resp.headers.get('content-type')})")); continue
            f.write_bytes(resp.content)
        text = "\n".join((p.extract_text() or "") for p in PdfReader(str(f)).pages[:6])
        d = extract(text)
        if d:
            rows.append({"date": r.date.date(), **d})
        else:
            fails.append((r.date.date(), "no EUR IRS row"))
    except Exception as e:
        fails.append((r.date.date(), f"{type(e).__name__}: {e}"[:120]))
    if k % 25 == 0:
        print(k, r.date.date(), "ok" if rows and rows[-1]["date"] == r.date.date() else "FAIL", flush=True)
    time.sleep(0.3)

out = pd.DataFrame(rows).sort_values("date")
cols = ["date"] + sorted([c for c in out.columns if c != "date"], key=lambda c: int(c[:-1]))
out[cols].to_csv(out_csv, index=False)
print("DONE rows", len(out), "fails", len(fails))
for f_ in fails[:40]:
    print("  FAIL", *f_)
