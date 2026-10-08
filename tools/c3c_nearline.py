"""C3c (8 Oct): where does the PAINTED near touchline sit in pitch metres, according to each frame's pose?
For every picture (1280x720) with a pose, take painted white-line pixels (calcheck.line_mask) within `band` px of the
drawn near touchline, below the horizon, project them back onto the ground with that pose and report the y (across the
pitch) they land on. The model says y = 64 (106 x 64). If the paint lands at, say, 66 m on every frame, the drawn
near line is off because the camera base / pitch width is off for this ground, not because a pose is wrong.

    python tools/c3c_nearline.py results/qa/c3b [results/qa/c3b_sfk ...]   -> results/qa/c3c/nearline_<folder>.json
Works on the C3b pictures (thin 1 px yellow model lines drawn on them; yellow is neither grey nor blue) and on raw frames.
The painted near line is found with paint_mask (calcheck.line_mask misses it: in shade it is sky-blue and wide)."""
import os, sys, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import lines as LN
from ipanema.calcheck import line_mask

def paint_mask(img):
    """painted lines incl. lines in deep shade, which read sky-blue, not white (Vallentuna's near touchline: hue ~100-110,
    saturation 160-240): bright against the grass around (31 px top-hat), bright, and grey or blue - not green."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, np.ones((31, 31), np.uint8)) > 25
    col = (hsv[..., 1] < 80) | ((hsv[..., 0] >= 95) & (hsv[..., 0] <= 125))
    return (th & (hsv[..., 2] > 150) & col).astype(np.uint8)

def backproject(camera, pose, uv, w=1280, h=720):
    """pixels (N,2) -> ground points (N,2) in pitch metres (NaN for rays that miss the ground in front)"""
    R = LN._rot(*pose[:3]) @ LN._base(*camera["base_tilt"]); f = pose[3] * w / 1280.0
    uv = np.asarray(uv, float); d = np.column_stack([(uv[:, 0] - w / 2) / f, (uv[:, 1] - h / 2) / f, np.ones(len(uv))]) @ R
    C = np.asarray(camera["C"], float)
    with np.errstate(divide="ignore", invalid="ignore"): t = -C[2] / d[:, 2]
    P = C[None, :2] + t[:, None] * d[:, :2]; P[~(t > 0)] = np.nan; return P

def near_line_pixels(img, camera, pose, band=120, W=LN.W_DEF):
    """painted-line pixels within band px of the drawn near touchline (the part > 20 m along it from nothing in particular:
    all of it in view); returns (uv (N,2), drawn distance (N,))"""
    h, w = img.shape[:2]; S = LN.projected_segments(camera, pose, w, h)[1]
    if not len(S): return np.zeros((0, 2)), np.zeros(0)
    m = paint_mask(img); m[: int(0.40 * h)] = 0                             # the near line is in the lower part; skip fence/sky
    ys, xs = np.nonzero(m)
    if not len(xs): return np.zeros((0, 2)), np.zeros(0)
    P = np.c_[xs, ys].astype(float); d = LN._pt_seg_dist(P, S); k = d <= band
    return P[k], d[k]

def measure(img, camera, pose, band=120):
    uv, d = near_line_pixels(img, camera, pose, band)
    if len(uv) < 60: return None
    G = backproject(camera, pose, uv); ok = np.isfinite(G).all(1) & (G[:, 0] > -3) & (G[:, 0] < 109) & (np.abs(G[:, 1] - 64) < 8)
    if ok.sum() < 60: return None
    G = G[ok]; C = np.asarray(camera["C"], float)[:2]; dist = np.linalg.norm(G - C, axis=1)
    y = G[:, 1]; med = float(np.median(y))
    keep = np.abs(y - med) < 1.5                                                # the one line, not socks / other paint
    if keep.sum() < 50: return None
    y, dist, x = y[keep], dist[keep], G[keep, 0]
    fit = np.polyfit(dist, y, 1) if np.ptp(dist) > 2 else [0.0, float(np.median(y))]
    return {"n": int(keep.sum()), "y_med": round(float(np.median(y)), 2), "y_iqr": round(float(np.subtract(*np.percentile(y, [75, 25]))), 2),
            "dist_m": [round(float(dist.min()), 1), round(float(dist.max()), 1)], "x_m": [round(float(x.min()), 1), round(float(x.max()), 1)],
            "slope_per_m": round(float(fit[0]), 3), "px_off_med": round(float(np.median(d[ok][keep])), 1)}

def main():
    out_dir = "results/qa/c3c"; os.makedirs(out_dir, exist_ok=True)
    for d in sys.argv[1:] or ["results/qa/c3b"]:
        S = json.load(open(f"{d}/sample.json")); G = json.load(open(f"{d}/grades.json")) if os.path.exists(f"{d}/grades.json") else {}
        rows = {}
        for q in S["rows"]:
            p = f"{d}/{q['id']}.jpg"
            if not os.path.exists(p): continue
            r = measure(cv2.imread(p), S["camera"], np.array(q["pose"], float))
            if r: r["grade"] = (G.get(q["id"]) or ["?"])[0]; r["t"] = q["t"]; rows[q["id"]] = r
        good = [r for r in rows.values() if r["grade"] == "g"]
        summ = {"frames": len(rows), "good_frames": len(good),
                "y_med_good": round(float(np.median([r["y_med"] for r in good])), 2) if good else None,
                "slope_med_good": round(float(np.median([r["slope_per_m"] for r in good])), 3) if good else None}
        json.dump({"folder": d, "camera": S["camera"], "summary": summ, "rows": rows}, open(f"{out_dir}/nearline_{os.path.basename(d)}.json", "w"), indent=1)
        print(d, summ)
        for k, r in sorted(rows.items()): print(" ", k, r)

if __name__ == "__main__": main()
