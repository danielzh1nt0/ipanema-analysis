"""CI step [trainset:<id>,...]: labels + review pictures per training match -> results/review/<match>/items.json"""
import os, re, json, traceback, modal
msg = os.environ.get("MSG", ""); ids = re.search(r"\[trainset:([A-Za-z0-9_.,-]+)\]", msg).group(1).split(",")
f = modal.Function.from_name("ipanema", "build_trainset"); summ = {}
for mid, r in zip(ids, f.map(ids, return_exceptions=True)):
    d = f"results/review/{mid}"; os.makedirs(d, exist_ok=True)
    if isinstance(r, Exception) or "error" in r: summ[mid] = {"error": repr(r)[:500] if isinstance(r, Exception) else r["error"]}; continue
    json.dump(r["items"], open(f"{d}/items.json", "w")); summ[mid] = {k: v for k, v in r.items() if k != "items"}; summ[mid]["review_items"] = len(r["items"])
json.dump(summ, open("results/review/summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))
