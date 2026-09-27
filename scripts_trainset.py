"""CI step [trainset:<id>,...]: labels + review pictures per training match -> results/review/<match>/items.json"""
import os, re, json, traceback, modal
msg = os.environ.get("MSG", ""); m = re.search(r"\[trainset(-clicks)?:([A-Za-z0-9_.,-]+)\]", msg)
src = "clicks" if m.group(1) else "wasb"; ids = m.group(2).split(",")
f = modal.Function.from_name("ipanema", "build_trainset"); summ = {}
for mid, r in zip(ids, f.starmap([(i, 28, 12, src) for i in ids], return_exceptions=True)):
    d = f"results/review/{mid}" + ("_clicks" if src == "clicks" else ""); os.makedirs(d, exist_ok=True)
    if isinstance(r, Exception) or "error" in r: summ[mid] = {"error": repr(r)[:500] if isinstance(r, Exception) else r["error"]}; continue
    json.dump(r["items"], open(f"{d}/items.json", "w")); summ[mid] = {k: v for k, v in r.items() if k != "items"}; summ[mid]["review_items"] = len(r["items"])
json.dump(summ, open(f"results/review/summary{'_clicks' if src == 'clicks' else ''}.json", "w"), indent=1); print(json.dumps(summ, indent=1))
