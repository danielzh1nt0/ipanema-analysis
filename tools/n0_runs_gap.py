"""N0 18 Oct: sprint list (metrics.runs) before/after the track-gap fix, rebuilt from the exported frames (10 fps) of the repo copies.
Usage: PYTHONPATH=. python tools/n0_runs_gap.py <old_metrics.py> -> results/qa/n0-18b/runs_gap.json"""
import sys, json, glob, os, importlib.util, numpy as np
from ipanema import metrics as NEW
spec = importlib.util.spec_from_file_location("metrics_old", sys.argv[1]); OLD = importlib.util.module_from_spec(spec)
sys.modules["metrics_old"] = OLD; OLD.__package__ = "ipanema"; spec.loader.exec_module(OLD)
res = {}
for d in sorted(glob.glob("results/volume/runs/matches/*/")):
    m = os.path.basename(d.rstrip("/")); chunks = sorted(glob.glob(f"{d}frames_*.json"))
    if not chunks: continue
    frames = [f for c in chunks for f in json.load(open(c))["frames"]]
    if len(frames) < 2: continue
    fps = round(1 / np.median(np.diff([f["t"] for f in frames[:200]])), 2)
    per, state = {}, []
    for k, f in enumerate(frames):
        per[k] = [(p["id"], p["team"], np.array(p["m"], float)) for p in f["players"] if p.get("m") and p.get("state") == "observed"]
        state.append({"A": 0, "B": 1}.get(f.get("possession"), 2))
    o, n = OLD.runs(per, fps, state, None), NEW.runs(per, fps, state, None)
    ok = {(r["player"], r["t"]) for r in o}; extra = [r for r in n if (r["player"], r["t"]) not in ok]
    res[m] = {"fps": fps, "minutes": round(len(frames) / fps / 60, 1), "runs_old": len(o), "runs_new": len(n), "old_all_kept": ok <= {(r["player"], r["t"]) for r in n},
              "new_median_s": round(float(np.median([r["t_end"] - r["t"] for r in extra])), 2) if extra else None,
              "new_median_speed": round(float(np.median([r["speed_ms"] for r in extra])), 1) if extra else None, "sample_new": extra[:12]}
    print(m, {k: v for k, v in res[m].items() if k != "sample_new"})
os.makedirs("results/qa/n0-18b", exist_ok=True); json.dump(res, open("results/qa/n0-18b/runs_gap.json", "w"), indent=1)
