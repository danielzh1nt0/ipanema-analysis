"""B1 (28 Sep): ball-with-player option in the picker, graded on the 34 checked SFK-BP moments. Free, local."""

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
feet_hit = sum(1 for i, g in gt.items() if new.get(i) and min(np.hypot(r[3][0]-g[0], r[3][1]-g[1]) for r in new[i]) <= 30)
print(f"moments where some player's feet are within 30 px of the ball: {feet_hit}/{len(gt)}")
nog = [i for i, g in gt.items() if not (cands.get(i) and min(np.hypot(x-g[0], y-g[1]) for x, y, _ in cands[i]) <= 30)]
nog_feet = sum(1 for i in nog if new.get(i) and min(np.hypot(r[3][0]-gt[i][0], r[3][1]-gt[i][1]) for r in new[i]) <= 30)
print(f"of the {len(nog)} moments with no finder guess on the ball, feet are on it in {nog_feet}")
dy = [gt[i][1] - min(new[i], key=lambda r: np.hypot(r[3][0]-gt[i][0], r[3][1]-gt[i][1]))[3][1] for i in gt if new.get(i)]
print("ball y minus nearest feet y (px), median:", float(np.median(dy)))
res = {}
base = BL.bridge(BL.pick_v2(cands, H, L, W, per=new, fps=fps, log=q), fps)
res["off"] = list(score(base)) + [len(base)]
for only in (False, True):
    for pc in (1.5, 2.0, 2.5, 2.8):
        b = BL.bridge(BL.pick_v2(cands, H, L, W, per=new, fps=fps, log=q, poss_cost=pc, poss_only_empty=only), fps)
        res[f"cost {pc}{' only when no guess' if only else ''}"] = list(score(b)) + [len(b)]
for k, v in res.items(): print(f"{k:32s} right {v[0]}/{v[1]}  frames with a ball {v[3]}/{n}", flush=True)
os.makedirs("results/picker", exist_ok=True); json.dump(res, open("results/picker/posslab.json", "w"), indent=1)
