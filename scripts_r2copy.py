"""CI step [r2copy:<id>,...]: copy training matches from the Modal volume to R2 -> results/datasets/r2copy.json"""
import os, re, json, modal
ids = re.search(r"\[r2copy:([A-Za-z0-9_.,-]+)\]", os.environ.get("MSG", "")).group(1).split(",")
f = modal.Function.from_name("ipanema", "copy_to_r2"); out = {}
for mid, r in zip(ids, f.map(ids, return_exceptions=True)): out[mid] = {"error": repr(r)[:500]} if isinstance(r, Exception) else r
os.makedirs("results/datasets", exist_ok=True); json.dump(out, open("results/datasets/r2copy.json", "w"), indent=1); print(json.dumps(out, indent=1))
