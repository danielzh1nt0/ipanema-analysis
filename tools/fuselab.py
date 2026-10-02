"""S8 (2 Oct): fusion weights RF-DETR vs WASB, overall and NEAR FEET only (a candidate within feet_px of a player's foot pixel),
graded through the picker on the SFK-BP keys (34, B4) with the Kaggle-recomputed old-finder candidates + the clip's WASB guesses.
Baseline row = today's fusion on those inputs (29 / 284).   PYTHONPATH=. python tools/fuselab.py"""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL
q = lambda *a: None; CLIP = "SFKBP1109_s1200"
on = lambda b, f, x, y: f in b and np.hypot(b[f][0] - x, b[f][1] - y) <= 30
P_ = pickle.load(open(f"results/volume/cache/{CLIP}/picker_inputs.pkl", "rb")); fps, H, L, W = P_["fps"], P_["H"], P_["L"], P_["W"]
per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P_["per"].items()}
feet = {k: np.array([r[3] for r in v if r[3] is not None]) for k, v in per.items()}
sgt = {int(k): v[:2] for k, v in json.load(open(f"results/volume/reference/{CLIP}/ball_gt.json")).items() if v}
b4 = [m for m in json.load(open("results/ball/b4/key.json"))["moments"] if m["verdict"] == "ball"]
def load(p):
    o = pickle.load(open(p, "rb")); o = o[0] if isinstance(o, tuple) else o
    return {int(k): [(float(a), float(b), float(c)) for a, b, c in v] for k, v in o.items()}
rf = load("results/kaggle/ballfinder_neg/cands_old_SFKBP1109_s1200.pkl"); wb = load(f"results/volume/cache/{CLIP}/ball_cands_wasb_1790008894_t2x2_thr0.05.pkl")
def fuse(wy=1.5, wbw=1.0, bonus=0.8, px=12, feet_px=None, wy_feet=None, wb_feet=None):
    out = {}
    for k in set(rf) | set(wb):
        ft = feet.get(k); nearf = lambda x, y: ft is not None and len(ft) and np.hypot(ft[:, 0] - x, ft[:, 1] - y).min() <= feet_px
        cands = []
        for x, y, c in rf.get(k, []):
            w = wy_feet if (feet_px and wy_feet is not None and nearf(x, y)) else wy; cands.append([x, y, w * c])
        for x, y, s in wb.get(k, []):
            w = wb_feet if (feet_px and wb_feet is not None and nearf(x, y)) else wbw
            j = next((i for i, z in enumerate(cands) if np.hypot(z[0] - x, z[1] - y) <= px), None)
            if j is not None: cands[j][2] += w * s + bonus
            else: cands.append([x, y, w * s])
        out[k] = [(x, y, min(0.99, sc / 2.0)) for x, y, sc in cands]
    return out
def grade(c):
    ball = BL.bridge(BL.pick_v2(c, H, L, W, per=per, fps=fps, log=q), fps)
    return int(sum(on(ball, f, *g) for f, g in sgt.items())), int(sum(on(ball, m["frame"], m["x"], m["y"]) for m in b4))
rows = []
for name, kw in (("today (1.5 / 1.0 / bonus 0.8)", {}), ("WASB 1.5", {"wbw": 1.5}), ("WASB 2.0", {"wbw": 2.0}), ("RF 1.0 WASB 1.5", {"wy": 1.0, "wbw": 1.5}),
                 ("near feet 40 px: WASB 2.0", {"feet_px": 40, "wb_feet": 2.0}), ("near feet 40 px: WASB 2.5 RF 1.0", {"feet_px": 40, "wb_feet": 2.5, "wy_feet": 1.0}),
                 ("near feet 60 px: WASB 2.0", {"feet_px": 60, "wb_feet": 2.0}), ("near feet 40 px: RF 2.5", {"feet_px": 40, "wy_feet": 2.5}), ("bonus 1.2", {"bonus": 1.2})):
    s34, sb4 = grade(fuse(**kw)); rows.append({"name": name, "kw": kw, "sfk34": s34, "b4": sb4}); print(f"{name:36s} sfk34 {s34}/34  b4 {sb4}/285", flush=True)
json.dump(rows, open("results/ball/fuselab_2026-10-02.json", "w"), indent=1)
