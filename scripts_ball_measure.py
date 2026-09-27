"""CI step: measure both ball finders on a short piece of a training match; report into results/datasets/. If the call
fails or times out, the progress log and partial result are read back from the volume instead."""
import os, re, json, base64, traceback, modal
msg = os.environ.get("MSG", ""); mid = re.search(r"\[ballmeasure:([A-Za-z0-9_.-]+)\]", msg).group(1)
os.makedirs("results/datasets", exist_ok=True); START, DUR = 1200, 120
try: out = modal.Function.from_name("ipanema", "ball_measure").remote(mid, START, DUR, 3, False)
except Exception:
    out = {"call_error": traceback.format_exc()[-1500:]}
    f = modal.Function.from_name("ipanema", "fetch_file")
    for name in ("ball_measure.json", "ball_measure.log"):
        try:
            r = f.remote(f"logs/{mid}_s{START}_d{DUR}/{name}", 0)
            out[name] = base64.b64decode(r["b64"]).decode(errors="replace")[-4000:] if "b64" in r else r
        except Exception as e: out[name] = repr(e)
json.dump(out, open(f"results/datasets/ball_measure_{mid}.json", "w"), indent=1); print(json.dumps(out, indent=1))
