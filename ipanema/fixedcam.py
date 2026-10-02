"""3 Oct: a FIXED camera (tactical / broadcast clip, no pan): one pitch->pixel homography for every frame, from pitch points
read off one frame by hand. Spec: calibration/fixed/<match>.json =
  {"pitch": {"length": 105, "width": 68}, "image_size": [1920, 1080],
   "points": [{"pitch_m": [x, y], "px": [u, v], "name": "..."}, ...]}   (>= 6 points, spread over the pitch)
Pitch metres: x along the length (0 = left goal line as seen), y across (0 = top touchline as seen). The homography is fitted
with RANSAC; points more than max_err_px off are reported. A clip of another size is scaled from image_size."""
import os, re, json, numpy as np, cv2

def find(match_id, root=None):
    d = root or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration", "fixed")
    mid = re.sub(r"(_c\d{3}|_s\d+(_d\d+)?)$", "", match_id)
    for name in (match_id, mid):
        p = os.path.join(d, f"{name}.json")
        if os.path.exists(p): return p
    return None

def fit(spec, w, h):
    """-> (H pitch->pixel 3x3, per-point error px)"""
    iw, ih = spec["image_size"]; sx, sy = w / iw, h / ih
    P = np.float32([p["pitch_m"] for p in spec["points"]]); Q = np.float32([[p["px"][0] * sx, p["px"][1] * sy] for p in spec["points"]])
    if len(P) < 4: raise ValueError("need at least 4 points")
    H, _ = cv2.findHomography(P, Q, cv2.RANSAC, 8.0) if len(P) >= 6 else (cv2.getPerspectiveTransform(P[:4], Q[:4]), None)
    proj = cv2.perspectiveTransform(P.reshape(-1, 1, 2), H).reshape(-1, 2); err = np.linalg.norm(proj - Q, axis=1)
    return H.astype(np.float64), err

def calibration(path, n_frames, w, h, max_err_px=12.0, log=print):
    spec = json.load(open(path)); L, W = float(spec["pitch"]["length"]), float(spec["pitch"]["width"])
    H, err = fit(spec, w, h); bad = [(p.get("name", i), round(float(e), 1)) for i, (p, e) in enumerate(zip(spec["points"], err)) if e > max_err_px]
    log(f"calibration: fixed camera from {len(spec['points'])} hand-read points ({os.path.basename(path)}): median error {np.median(err):.1f} px, max {err.max():.1f} px" + (f"; off by > {max_err_px:.0f} px: {bad}" if bad else ""))
    return {"coverage": 1.0, "frozen": 0, "H": {k: H for k in range(n_frames)}, "L": L, "W": W, "unsure": set(), "fixed": True}
