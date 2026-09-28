"""CI step [statreview:<match>]: stoppage + possession pictures for Daniel's answer key -> results/review/stats_<match>/"""
import os, re, json, modal
mid = re.search(r"\[statreview:([A-Za-z0-9_.-]+)\]", os.environ.get("MSG", "")).group(1); d = f"results/review/stats_{mid}"; os.makedirs(d, exist_ok=True)
try: r = modal.Function.from_name("ipanema", "stat_review").remote(mid)
except Exception as e: r = {"error": repr(e)[:1500]}
if "items" in r: json.dump(r["items"], open(f"{d}/items.json", "w"))
json.dump({k: v for k, v in r.items() if k != "items"}, open(f"{d}/summary.json", "w"), indent=1); print({k: v for k, v in r.items() if k != "items"})
