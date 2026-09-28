"""CI step [playerbench:<id>,...]: player detection pictures per match -> results/qa/players/<match>/ (CPU)"""
import os, re, json, base64, modal
ids = re.search(r"\[playerbench:([A-Za-z0-9_.,-]+)\]", os.environ.get("MSG", "")).group(1).split(",")
f = modal.Function.from_name("ipanema", "player_bench"); summ = {}
for mid, r in zip(ids, f.map(ids, return_exceptions=True)):
    d = f"results/qa/players/{mid}"; os.makedirs(d, exist_ok=True)
    if isinstance(r, Exception) or "error" in r: summ[mid] = {"error": repr(r)[:800] if isinstance(r, Exception) else r["error"]}; continue
    for it in r["items"]: open(f"{d}/f{it['frame']:07d}.jpg", "wb").write(base64.b64decode(it.pop("img")))
    json.dump(r["items"], open(f"{d}/boxes.json", "w")); summ[mid] = {"frames": r["frames"], "seconds": r["seconds"]}
json.dump(summ, open("results/qa/players/summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))
