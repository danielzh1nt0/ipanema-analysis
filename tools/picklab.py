"""28 Sep: ball picker offline on the SFK-BP clip (saved WASB guesses, results/picker/*.npz), graded on the 34 checked moments.
Compares: old players vs new players (Modal full-clip run, rows_all.json.gz) and picker variants. Free, local."""
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
res = {}
for name, per, kw in (("v2, old players", old, {}), ("v2, new players", new, {}),
                      ("v2, new players, near weight 2", new, {"near_w": 2.0}), ("v2, new players, near weight 3", new, {"near_w": 3.0})):
    b = BL.bridge(BL.pick_v2(cands, H, L, W, per=per, fps=fps, log=q, **kw), fps); res[name] = score(b); print(f"{name:34s} {res[name][0]}/{res[name][1]} right (best possible {res[name][2]})", flush=True)
json.dump(res, open("results/picker/picklab.json", "w"), indent=1)

# which moments fail, and why
b = BL.bridge(BL.pick_v2(cands, H, L, W, per=new, fps=fps, log=q), fps); why = {"right": 0, "no guess on the ball": 0, "picked another guess": 0, "no pick": 0}
for i, g in sorted(gt.items()):
    has = cands.get(i) and min(np.hypot(x - g[0], y - g[1]) for x, y, _ in cands[i]) <= 30
    if i in b and np.hypot(b[i][0] - g[0], b[i][1] - g[1]) <= 30: why["right"] += 1
    elif not has: why["no guess on the ball"] += 1
    elif i not in b: why["no pick"] += 1
    else: why["picked another guess"] += 1
print(why)

def carry_fill(ball, per, H, fps, max_gap_s=4.0, near_px=45.0):
    """Veo-style: when the ball disappears at a player's feet and reappears at the SAME player's feet, he carried it:
    place it at his feet for the gap. Pixel distances on the frame (feet = player px)."""
    out = dict(ball); ks = sorted(ball)
    def owner(k):
        b = np.array(ball[k]); best = None
        for r in per.get(k, []):
            dd = np.linalg.norm(np.asarray(r[3]) - b)
            if dd < near_px and (best is None or dd < best[0]): best = (dd, r[0])
        return None if best is None else best[1]
    filled = 0
    for a, c in zip(ks, ks[1:]):
        if 1 < c - a <= max_gap_s * fps:
            oa, oc = owner(a), owner(c)
            if oa is not None and oa == oc:
                for k in range(a + 1, c):
                    r = next((r for r in per.get(k, []) if r[0] == oa), None)
                    if r is not None: out[k] = [float(r[3][0]), float(r[3][1])]; filled += 1
    return out, filled
raw = BL.pick_v2(cands, H, L, W, per=new, fps=fps, log=q)
for gap in (2.0, 4.0, 6.0):
    cf, nf = carry_fill(raw, new, H, fps, gap); b2 = BL.bridge(cf, fps)
    print(f"v2 + carry fill (gaps up to {gap:.0f} s, {nf} frames filled): {score(b2)[0]}/{len(gt)}; frames with a ball {len(b2)}/{n} (was {len(BL.bridge(raw, fps))})")
