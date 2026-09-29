"""29 Sep: the app ball picker (v2) on the 34 app moments with the new RF-DETR ball guesses: alone, fused with WASB, and
today's app setup (WASB + click model) for comparison. Free, local. Input: results/picker/ball_cands_rfdetr_<clip>.json.gz."""
import sys, os, json, gzip, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, linecal as LC
M = "SFKBP1109_s1200"
d = np.load(f"results/picker/{M}.npz", allow_pickle=True); n = int(d["n"]); fps = float(d["fps"])
gt = {int(k): v for k, v in json.loads(bytes(d["gt"]).decode()).items() if v is not None}
rows = LC.find_rows(".", M); cal = LC.calibration_for_clip(rows[2], n, fps, 1920, 1080, offset_s=1200, log=lambda *a: None); H = cal["H"]; L, W = cal["L"], cal["W"]
def to_m(Hk, px): v = np.linalg.inv(Hk) @ np.array([px[0], px[1], 1.0]); return v[:2] / v[2]
R = json.load(gzip.open("results/qa/tracktest_gpu/rows_all.json.gz", "rt"))["rows"]["new, RF-DETR"]
per = {int(k): [[pid, t, to_m(H[int(k)], px), np.array(px), None, False] for pid, t, px, fl in v if px is not None] for k, v in R.items()}
C = f"results/volume/cache/{M}/"
def load(f):
    o = pickle.load(open(C + f, "rb")); o = o[0] if isinstance(o, tuple) else o
    return {int(k): [(float(a), float(b), float(c)) for a, b, c in v] for k, v in o.items()}
clk = load("ball_cands_clicks_round_20260928_0108.pkl"); w30 = load("ball_cands_wasb_1790008894_t2x2_thr0.05_lp30.pkl")
rf = {}
for k, x, y, c in json.load(gzip.open(f"results/picker/ball_cands_rfdetr_{M}.json.gz", "rt")): rf.setdefault(int(k), []).append((x, y, c))
def score(ball, cands, hit=30):
    ok = sum(1 for i, g in gt.items() if i in ball and np.hypot(ball[i][0] - g[0], ball[i][1] - g[1]) <= hit)
    ceil = sum(1 for i, g in gt.items() if cands.get(i) and min(np.hypot(x - g[0], y - g[1]) for x, y, _ in cands[i]) <= hit)
    return ok, ceil
q = lambda *a: None; res = {}
for name, cd in (("today: WASB + click model", BL.fuse_candidates(clk, w30)), ("RF-DETR alone", rf),
                 ("RF-DETR + WASB", BL.fuse_candidates(rf, w30)), ("RF-DETR (x2) + WASB", BL.fuse_candidates(rf, w30, wy=3.0))):
    b = BL.bridge(BL.pick_v2(cd, H, L, W, per=per, fps=fps, log=q), fps); ok, ceil = score(b, cd)
    res[name] = {"right": ok, "of": len(gt), "ceiling": ceil}; print(f"{name:28s} {ok}/{len(gt)}  (ball among guesses {ceil})", flush=True)
json.dump(res, open("results/ball/rf_picker.json", "w"), indent=1)
