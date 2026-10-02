"""Z1 (3 Oct): a ball in the air projects off the pitch plane; the picker treats off-pitch candidates as clutter (margin 1.5 m,
air_w 0.6). Sweep margin / air_w on the exact app inputs; keep only if AIK and SFK-BP keys do not drop. Free, local."""
import sys, os, json, pickle, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL
q = lambda *a: None
on = lambda b, f, x, y: f in b and np.hypot(b[f][0] - x, b[f][1] - y) <= 30
def load(m):
    P = pickle.load(open(f"results/volume/cache/{m}/picker_inputs.pkl", "rb"))
    P["per"] = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P["per"].items()}
    return P
def pick(P, kw): return BL.bridge(BL.pick_v2(P["cands"], P["H"], P["L"], P["W"], per=P["per"], fps=P["fps"], log=q, **kw), P["fps"])
A, S = load("p15u-vs-aik-2026-09-21-bd09_s2520"), load("SFKBP1109_s1200")
agt = {int(k): v for k, v in json.load(open("reference/p15u-vs-aik-2026-09-21-bd09_s2520/ball_gt.json")).items()}
sgt = {int(k): v for k, v in json.load(open("results/volume/reference/SFKBP1109_s1200/ball_gt.json")).items() if v}
b4 = [m for m in json.load(open("results/ball/b4/key.json"))["moments"] if m["verdict"] == "ball"]
def score(kw):
    a = pick(A, kw); s = pick(S, kw)
    return {"aik": int(sum(on(a, f, *g) for f, g in agt.items())), "sfk34": int(sum(on(s, f, *g[:2]) for f, g in sgt.items())), "b4": int(sum(on(s, m["frame"], m["x"], m["y"]) for m in b4))}
res = []
for margin, air_w in itertools.product((1.5, 4.0, 8.0), (0.6, 0.3, 1.0)):
    kw = {"margin": margin, "air_w": air_w}; r = {"kw": kw, **score(kw)}; res.append(r); print(r, flush=True)
json.dump(res, open("results/ball/airlab_2026-10-03.json", "w"), indent=1)
