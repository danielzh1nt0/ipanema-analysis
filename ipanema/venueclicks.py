"""New venue from Daniel's pitch-point clicks alone (26 Sep): camera base + per-frame poses solved jointly, no prior.

Input: {frame name: {"pairs": [[point name, u, v], ...]}} in 1280x720 pixels (the click page). Frames with fewer than
`min_pairs` points are ignored. Output: the same solution format as calibration/panorama/<match>_clicks_solution.json,
so basefix / lines / matchcal work unchanged. Self-check: solve on half the frames, report click error on the other half."""
import json, numpy as np
from scipy.optimize import least_squares
from . import lines as LN
from .label import pitch_keypoints

def _obs(clicks, kp, min_pairs):
    out = []
    for name in sorted(clicks):
        pr = [(kp[n], (u, v)) for n, u, v in clicks[name].get("pairs", []) if n in kp and "penalty spot" not in n]
        if len(pr) >= min_pairs: out.append((name, np.array([p for p, _ in pr], float), np.array([q for _, q in pr], float)))
    return out

def _first_pose(cam, P, Q, w=1280, h=720):
    """coarse grid over pan/tilt/f, best click error; then a local least squares"""
    best = (np.inf, None)
    for pan in np.radians(np.arange(-180, 0, 4.0)):
        for tilt in np.radians(np.arange(2, 40, 3.0)):
            for f in (700, 1000, 1400, 2000, 2800, 4000):
                q = LN.project(cam, [pan, tilt, 0.0, f], P, w, h); e = np.nan_to_num(np.linalg.norm(q - Q, axis=1), nan=1e4).mean()
                if e < best[0]: best = (e, [pan, tilt, 0.0, f])
    r = least_squares(lambda p: np.nan_to_num(LN.project(cam, p, P, w, h) - Q, nan=1e3).ravel(), best[1], x_scale=[1e-2, 1e-2, 1e-2, 50.0], max_nfev=200)
    return [float(v) for v in r.x]

def solve(clicks, L=106.0, W=64.0, min_pairs=4, w=1280, h=720, log=print, C0=None):
    kp = pitch_keypoints(L, W); obs = _obs(clicks, kp, min_pairs)
    if len(obs) < 3: raise ValueError(f"need at least 3 frames with {min_pairs}+ clicked points, have {len(obs)}")
    starts = [C0] if C0 is not None else [[L / 2, W + dy, -z] for dy in (3.0, 6.0, 10.0) for z in (4.0, 6.0)]
    best = None
    for C in starts:
        cam = {"C": list(C), "base_tilt": [0.0, 0.0]}; poses = [_first_pose(cam, P, Q, w, h) for _, P, Q in obs]
        x0 = np.r_[C, 0.0, 0.0, np.array(poses).ravel()]
        def unpack(x): return {"C": list(x[:3]), "base_tilt": list(x[3:5])}, x[5:].reshape(-1, 4)
        def res(x):
            cam, Pp = unpack(x); return np.concatenate([np.nan_to_num(LN.project(cam, p, P, w, h) - Q, nan=500).ravel() for (_, P, Q), p in zip(obs, Pp)])
        r = least_squares(res, x0, loss="soft_l1", f_scale=5.0, x_scale="jac", max_nfev=4000)
        cost = float(np.median(np.abs(res(r.x))))
        if log: log(f"  start {np.round(C, 1)} -> base {np.round(r.x[:3], 2)}, median residual {cost:.2f} px")
        if best is None or cost < best[0]: best = (cost, r.x)
    cam, Pp = unpack(best[1])
    frames = [{"session": "venue", "frame": name, "pose": [float(v) for v in p]} for (name, _, _), p in zip(obs, Pp)]
    errs = {name: float(np.median(np.linalg.norm(LN.project(cam, p, P, w, h) - Q, axis=1))) for (name, P, Q), p in zip(obs, Pp)}
    sol = {"camera": cam, "image_size": [w, h], "pitch": [L, W], "frames": frames, "click_errors_px": errs,
           "note": "camera base + per-frame poses solved jointly from Daniel's pitch-point clicks, no prior"}
    return sol

def heldout(clicks, L=106.0, W=64.0, min_pairs=4, log=print):
    """solve on frames A (every other), place frames B with that base from their own clicks: B's click error is the honest number"""
    kp = pitch_keypoints(L, W); names = [n for n, _, _ in _obs(clicks, kp, min_pairs)]
    A = {n: clicks[n] for n in names[0::2]}; B = _obs({n: clicks[n] for n in names[1::2]}, kp, min_pairs)
    sa = solve(A, L, W, min_pairs, log=None); errs = []
    for name, P, Q in B:
        p = _first_pose(sa["camera"], P, Q); errs += list(np.linalg.norm(LN.project(sa["camera"], p, P, 1280, 720) - Q, axis=1))
    out = {"solved_on": len(A), "tested_on": len(B), "median_px": round(float(np.median(errs)), 1), "p90_px": round(float(np.percentile(errs, 90)), 1), "base_A": [round(v, 2) for v in sa["camera"]["C"]]}
    if log: log(f"  held-out: base from {len(A)} frames places the other {len(B)} frames' clicks at median {out['median_px']} px (p90 {out['p90_px']})")
    return out
