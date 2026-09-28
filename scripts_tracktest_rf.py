"""CI step [tracktest-rf:<start_s>:<dur_s>]: tracking comparison on a Modal GPU -> results/qa/tracktest_gpu/"""
import os, re, json, base64, modal
m = re.search(r"\[tracktest-rf:([0-9.]+):([0-9.]+)\]", os.environ.get("MSG", "")); st, du = float(m.group(1)), float(m.group(2))
OUT = "results/qa/tracktest_gpu"; os.makedirs(OUT, exist_ok=True)
try: r = modal.Function.from_name("ipanema", "tracktest_rf").remote(st, du)
except Exception as e: r = {"error": repr(e)[:2000], "files": {}}
for n, b in r.pop("files", {}).items(): open(f"{OUT}/{n}", "wb").write(base64.b64decode(b))
json.dump(r, open(f"{OUT}/run.json", "w"), indent=1); print(json.dumps(r, indent=1)[:3000])
