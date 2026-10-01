"""1 Oct (M1): speed-layer lab on SFK-BP's exact app inputs, graded on the 28 by-eye moments
(results/review/speedcheck_answers.json). Tries: smoothing the calibration over time before turning feet into metres
(the follow-cam calibration wobbles frame to frame), and the speed baseline. Free, local."""
import sys, os, json, pickle, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import motion as MO
from ipanema.calibration import to_m
P = pickle.load(open("results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl", "rb")); fps = P["fps"]; per = P["per"]; H = P["H"]; L, W = P["L"], P["W"]
G = json.load(open("results/review/speedcheck_answers.json"))["graded"]; M = {str(m["n"]): m for m in json.load(open("results/review/speedcheck_moments.json"))}
OK = {"stand": (0, 4), "walk": (2, 8), "jog": (6, 16), "run": (12, 30)}
GRID = np.array([[x, y] for x in np.linspace(0, L, 9) for y in np.linspace(0, W, 5)], float)
def smooth_H(w):
    if w == 0: return H
    ks = sorted(H); pix = np.array([cv2.perspectiveTransform(GRID.reshape(-1, 1, 2).astype(np.float64), H[k]).reshape(-1, 2) for k in ks])
    cs = np.concatenate([np.zeros((1,) + pix.shape[1:]), np.cumsum(pix, 0)]); out = {}
    for i, k in enumerate(ks):
        lo, hi = max(0, i - w), min(len(ks), i + w + 1); sp = (cs[hi] - cs[lo]) / (hi - lo)
        Hs, _ = cv2.findHomography(GRID, sp); out[k] = Hs if Hs is not None else H[k]
    return out
def grade(per2, base):
    MO.STEP_S = base; mo = MO.compute(per2, fps); ok = 0; rows = []
    for n, g in G.items():
        if g == "bad": continue
        m = M[n]; v = mo.get(m["frame"], {}).get(m["id"]); kmh = v[0] if v else None
        hit = kmh is not None and OK[g][0] <= kmh <= OK[g][1]; ok += hit; rows.append((n, g, kmh))
    sp = np.array([v[0] for f in mo.values() for v in f.values() if v[0] is not None])
    return ok, len(rows), rows, np.round(np.percentile(sp, [50, 95]), 1), round(100 * float((sp > 25).mean()), 2)
for w in (0, 8, 15, 30):
    Hs = smooth_H(w)
    per2 = {k: [[r[0], r[1], (to_m(Hs[k], [r[3]])[0] if r[3] is not None else r[2]), r[3]] for r in rows] for k, rows in per.items()}
    for base in (0.2, 0.5, 1.0):
        ok, n, rows, pct, hi = grade(per2, base)
        print(f"H smooth ±{w} frames, speed over {base}s: {ok}/{n} in range | km/h p50,p95 {pct} | >25 km/h {hi}%", flush=True)
        if base == 0.5: print("   ", [(a, b, c) for a, b, c in rows])
