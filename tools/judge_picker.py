"""29 Sep: the app ball picker with the judge blended into each guess score, graded on the 34 app moments. Free, local."""


import sys, os, json, gzip, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, linecal as LC
d = np.load("results/picker/SFKBP1109_s1200.npz", allow_pickle=True); n = int(d["n"]); fps = float(d["fps"])
cands = {}
for f, x, y, c in d["cands"]: cands.setdefault(int(f), []).append((float(x), float(y), float(c)))
gt = {int(k): v for k, v in json.loads(bytes(d["gt"]).decode()).items() if v is not None}
rows = LC.find_rows(".", "SFKBP1109_s1200"); cal = LC.calibration_for_clip(rows[2], n, fps, 1920, 1080, offset_s=1200, log=lambda *a: None); H = cal["H"]; L, W = cal["L"], cal["W"]
def to_m(Hk, px): v = np.linalg.inv(Hk) @ np.array([px[0], px[1], 1.0]); return v[:2] / v[2]
old = {}
for f, pid, team, mx, my, px, py, gk in d["players"]:
    old.setdefault(int(f), []).append([int(pid), "A" if team else "B", to_m(H[int(f)], (px, py)), np.array([px, py]), None, False])
R = json.load(gzip.open("results/qa/tracktest_gpu/rows_all.json.gz", "rt"))["rows"]["new, RF-DETR"]
new = {int(k): [[pid, t, to_m(H[int(k)], px), np.array(px), None, False] for pid, t, px, fl in v if px is not None] for k, v in R.items()}
def score(ball, hit=30):
    ok = sum(1 for i, g in gt.items() if i in ball and np.hypot(ball[i][0] - g[0], ball[i][1] - g[1]) <= hit)
    ceil = sum(1 for i, g in gt.items() if cands.get(i) and min(np.hypot(x - g[0], y - g[1]) for x, y, _ in cands[i]) <= hit)
    return ok, len(gt), ceil
q = lambda *a: None
import pickle
C = "results/volume/cache/SFKBP1109_s1200/"
def load(f):
    o = pickle.load(open(C + f, "rb")); o = o[0] if isinstance(o, tuple) else o
    return {int(k): [(float(a), float(b), float(c)) for a, b, c in v] for k, v in o.items()}
new_clk = load("ball_cands_clicks_round_20260928_0108.pkl"); w30 = load("ball_cands_wasb_1790008894_t2x2_thr0.05_lp30.pkl")
R = np.load("results/ball/scorer/judge_clip_scores.npz")["rows"]
J = {}
for k, x, y, s, p in R: J.setdefault(int(k), []).append((float(x), float(y), float(s), float(p)))
res = {}
for wj in (0.0, 0.5, 1.0, 2.0):
    wj_ = {int(k): [(x, y, min(0.99, (s + wj * p) / (1 + wj))) for x, y, s, p in v] for k, v in J.items()}
    for name, cd in (("wasb30", wj_), ("wasb30 + clicks", BL.fuse_candidates(new_clk, wj_))):
        b = BL.bridge(BL.pick_v2(cd, H, L, W, per=new, fps=fps, log=q), fps); r = score(b)
        res[f"{name}, judge weight {wj}"] = r[0]; print(f"{name:18s} judge weight {wj}: {r[0]}/{r[1]}", flush=True)
json.dump(res, open("results/ball/scorer/judge_picker.json", "w"), indent=1)
