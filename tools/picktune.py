"""1 Oct: tune the ball picker (pick_v2) for free, offline. Tuned on AIK (39-moment blind key, exact app inputs from
cache/<aik>/picker_inputs.pkl), guarded on SFK-BP (34 moments + B4 key 285 balls, local setup as tools/b4blab.py).
A setting is only kept if AIK goes up AND SFK-BP does not go down on either key. Writes results/ball/picktune.json."""
import sys, os, json, gzip, pickle, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, linecal as LC
q = lambda *a: None
on = lambda b, f, x, y: f in b and np.hypot(b[f][0] - x, b[f][1] - y) <= 30
# AIK: exact app inputs
A = "p15u-vs-aik-2026-09-21-bd09_s2520"
P = pickle.load(open(f"results/volume/cache/{A}/picker_inputs.pkl", "rb"))
aper = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P["per"].items()}
agt = {int(k): v for k, v in json.load(open(f"reference/{A}/ball_gt.json")).items()}
def aik(kw): b = BL.bridge(BL.pick_v2(P["cands"], P["H"], P["L"], P["W"], per=aper, fps=P["fps"], log=q, **kw), P["fps"]); return sum(on(b, f, *g) for f, g in agt.items())
# SFK-BP: local setup (as b4blab)
M = "SFKBP1109_s1200"
d = np.load(f"results/picker/{M}.npz", allow_pickle=True); n = int(d["n"]); fps = float(d["fps"])
sgt = {int(k): v for k, v in json.loads(bytes(d["gt"]).decode()).items() if v is not None}
rows = LC.find_rows(".", M); cal = LC.calibration_for_clip(rows[2], n, fps, 1920, 1080, offset_s=1200, log=q); H = cal["H"]; L, W = cal["L"], cal["W"]
def to_m(Hk, px): v = np.linalg.inv(Hk) @ np.array([px[0], px[1], 1.0]); return v[:2] / v[2]
R = json.load(gzip.open("results/qa/tracktest_gpu/rows_all.json.gz", "rt"))["rows"]["new, RF-DETR"]
sper = {int(k): [[pid, t, to_m(H[int(k)], px), np.array(px), None, False] for pid, t, px, fl in v if px is not None] for k, v in R.items()}
o = pickle.load(open(f"results/volume/cache/{M}/ball_cands_wasb_1790008894_t2x2_thr0.05_lp30.pkl", "rb")); o = o[0] if isinstance(o, tuple) else o
w30 = {int(k): [(float(a), float(b), float(c)) for a, b, c in v] for k, v in o.items()}
rf = {}
for k, x, y, c in json.load(gzip.open(f"results/picker/ball_cands_rfdetr_{M}.json.gz", "rt")): rf.setdefault(int(k), []).append((x, y, c))
scd = BL.fuse_candidates(rf, w30)
b4 = [m for m in json.load(open("results/ball/b4/key.json"))["moments"] if m["verdict"] == "ball"]
def sfk(kw):
    b = BL.bridge(BL.pick_v2(scd, H, L, W, per=sper, fps=fps, log=q, **kw), fps)
    return sum(on(b, i, *g) for i, g in sgt.items()), sum(on(b, m["frame"], m["x"], m["y"]) for m in b4)
base = {"aik": aik({}), "sfk34_b4": sfk({})}; print("default", base, flush=True)
grid = {"conf_w": [1.5, 2.5, 4.0], "near_w": [1.0, 0.5, 2.0], "jump_cost": [6.0, 3.0], "miss_cost": [3.0, 5.0]}
res = []
for vals in itertools.product(*grid.values()):
    kw = dict(zip(grid, vals))
    a = aik(kw); r = {"kw": kw, "aik": a}
    if a > base["aik"]: r["sfk34_b4"] = sfk(kw)
    res.append(r); print(r, flush=True)
ok = [r for r in res if "sfk34_b4" in r and r["sfk34_b4"][0] >= base["sfk34_b4"][0] and r["sfk34_b4"][1] >= base["sfk34_b4"][1]]
ok.sort(key=lambda r: (-r["aik"], -r["sfk34_b4"][0], -r["sfk34_b4"][1]))
json.dump({"default": base, "of": {"aik": len(agt), "sfk34": len(sgt), "b4": len(b4)}, "grid": grid, "passing": ok, "all": res}, open("results/ball/picktune.json", "w"), indent=1, default=int)
print("best passing:", ok[:3])
