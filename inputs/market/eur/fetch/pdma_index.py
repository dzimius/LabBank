"""Index the PDMA daily market report archive -> CSV (date, title, page_url)."""
import re, sys, time, csv
import truststore; truststore.inject_into_ssl()
import requests
from datetime import datetime

out = sys.argv[1]
H = {"User-Agent": "Mozilla/5.0 (LabBank research; contact via repo owner)"}
BASE = "https://www.pdma.gr"
LIST = BASE + "/en/investor-relations/daily-market-report"
item = re.compile(r'<h2[^>]*>\s*<a[^>]*href="(/en/investor-relations/daily-market-report/[^"?#]+)"[^>]*>\s*([^<]*?)\s*</a>\s*</h2>.*?<time datetime="(\d{4}-\d{2}-\d{2})', re.S)
rows, seen = [], set()
start, empty = 0, 0
while start <= 2300 and empty < 3:
    try:
        r = requests.get(LIST, params={"start": start} if start else None, headers=H, timeout=60)
        html = r.text
    except Exception as e:
        print("ERR", start, e); time.sleep(3); continue
    found = 0
    for url, title, d in item.findall(html):
        if url in seen: continue
        seen.add(url); found += 1
        rows.append((d, title.strip(), BASE + url))
    empty = empty + 1 if found == 0 else 0
    if start % 250 == 0: print(start, len(rows), rows[-1][0] if rows else None, flush=True)
    start += 5
    time.sleep(0.25)
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["date", "title", "page_url"]); w.writerows(sorted(rows))
print("DONE", len(rows), min(r[0] for r in rows), max(r[0] for r in rows))
