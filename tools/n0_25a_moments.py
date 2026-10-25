"""N0-25a: picture moments (for tools/n0_20b_frames.py on the free runner) of every run the new rule finds on the two app
clips: tiles every 0.33 s from 0.7 s before the run to 0.3 s after, the runner's feet as the red ring. Grade by eye:
is the ringed player sprinting (clearly running fast, the same person all through)? Ids are shuffled; which rule found
what is in moments_key.json, not on the sheet.
    PYTHONPATH=. python tools/n0_25a_moments.py"""
import os, sys, json, pickle, random, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import metrics as M

CLIPS = {"sfk": ("SFKBP1109_s1200", "results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl"),
         "aik": ("p15u-vs-aik-2026-09-21-bd09_s2520", "results/volume/cache/p15u-vs-aik-2026-09-21-bd09_s2520/picker_inputs.pkl")}
AR = {"A": True, "B": False}
OUT = "results/qa/n0-25a"


def px_of(per, tid, k):
    for j in range(k, k - 8, -1):
        for r in per.get(j, []):
            if r[0] == tid and r[3] is not None: return [float(r[3][0]), float(r[3][1])]
    return None


def build(short, clip, path, seed=25):
    d = pickle.load(open(path, "rb")); per = d["per"]; fps = d["fps"]; per = per if isinstance(per, dict) else dict(enumerate(per))
    new = M.runs(per, fps, np.full(len(per), 2), AR, smooth=True); old = M.runs(per, fps, np.full(len(per), 2), AR, smooth=False)
    rs = [("new", r) for r in new] + [("old", r) for r in old if not any(n["player"] == r["player"] and n["t"] <= r["t_end"] and r["t"] <= n["t_end"] for n in new)]
    random.Random(seed).shuffle(rs); moments, key = [], {}
    for i, (rule, r) in enumerate(rs):
        mid = f"{short}{i:02d}"; a = int((r["t"] - 0.7) * fps); b = int((r["t_end"] + 0.3) * fps); step = max(1, int(round(fps / 3)))
        tiles = [{"frame": k, "t": round(k / fps, 2), "ball_px": px_of(per, r["player"], k)} for k in range(max(0, a), b + 1, step)]
        moments.append({"id": mid, "clip": clip, "team": f"ring = player {r['player']}", "t": r["t"], "frame": int(r["t"] * fps), "attacks": "-", "tiles": tiles})
        key[mid] = {"rule": rule, **{k: r[k] for k in ("player", "team", "t", "t_end", "distance_m", "speed_ms")}}
    json.dump({"clip": clip, "moments": moments}, open(f"{OUT}/moments_{short}.json", "w"), indent=1)
    return key


if __name__ == "__main__":
    key = {}
    for short, (clip, path) in CLIPS.items(): key.update(build(short, clip, path))
    json.dump(key, open(f"{OUT}/moments_key.json", "w"), indent=1)
    print(len(key), "moments:", {r: sum(v["rule"] == r for v in key.values()) for r in ("new", "old")})
