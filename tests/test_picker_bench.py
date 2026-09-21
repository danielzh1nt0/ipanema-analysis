"""On the real SFK-BP test clip data, picker v2 must keep at least 24/34 and v1 must reproduce 18/34 (verified 21 Sep)."""
import os, sys, json, numpy as np, bisect
from ipanema import ball as BL, tracking as TR
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load():
    z = np.load(f"{ROOT}/results/picker/SFKBP1109_s1200.npz"); n = int(z["n"]); fps = float(z["fps"])
    cands = {k: [] for k in range(n)}
    for k, x, y, c in z["cands"]: cands[int(k)].append((float(x), float(y), float(c)))
    per = {k: [] for k in range(n)}
    for k, tid, team, mx, my, fx, fy, gk in z["players"]:
        per[int(k)].append([int(tid), "A" if team == 0 else "B", np.array([mx, my], float), np.array([fx, fy], float), np.zeros(4), bool(gk)])
    hk = list(z["h_frames"]); Hm = {int(k): z["h"][i].astype(np.float64) for i, k in enumerate(hk)}; valid = sorted(Hm); H = {}
    for k in range(n):
        if k in Hm: H[k] = Hm[k]
        else:
            j = bisect.bisect_left(valid, k); c = [valid[x] for x in (j - 1, j) if 0 <= x < len(valid)]; H[k] = Hm[min(c, key=lambda v: abs(v - k))]
    gt = {int(k): v for k, v in json.loads(bytes(z["gt"]).decode()).items()}
    per, _ = TR.clean(per, 120.0, 70.0, fps, log=lambda *a: None)
    return cands, per, H, fps, gt

def _score(ball, gt):
    return sum(1 for i, g in gt.items() if g is not None and i in ball and np.hypot(ball[i][0] - g[0], ball[i][1] - g[1]) <= 30)

def test_pickers_on_real_clip():
    cands, per, H, fps, gt = _load()
    v2 = BL.bridge(BL.pick_v2(cands, H, 120.0, 70.0, per=per, fps=fps, log=lambda *a: None), fps)
    g = BL.pick_global(cands, H, 120.0, 70.0, per=per, fps=fps, log=lambda *a: None)
    v1 = BL.bridge(g, fps)
    assert _score(v2, gt) >= 24, _score(v2, gt)
    assert _score(v1, gt) == 18, _score(v1, gt)
