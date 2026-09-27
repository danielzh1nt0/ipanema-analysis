"""CI step: measure both ball finders on 10 minutes of a training match; report into results/datasets/"""
import os, re, json, traceback, modal
msg = os.environ.get("MSG", ""); mid = re.search(r"\[ballmeasure:([A-Za-z0-9_.-]+)\]", msg).group(1)
os.makedirs("results/datasets", exist_ok=True)
try: out = modal.Function.from_name("ipanema", "ball_measure").remote(mid, 1200, 600)
except Exception: out = {"error": traceback.format_exc()[-3000:]}
json.dump(out, open(f"results/datasets/ball_measure_{mid}.json", "w"), indent=1); print(json.dumps(out, indent=1))
