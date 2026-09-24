"""Camera base check/fix from the painted near touchline (24 Sep).

Daniel's clicks are all far from the camera, so they pin the base only loosely in the near field: on every frame where
the near touchline is visible, the drawn line sits on the pitch side of the painted one (seen on 6 frames, both ends).
Here the painted near touchline is found automatically in each clicked frame (white-line pixels in a band around the
drawn line, one straight line by RANSAC), and the base is re-solved with those lines added to the clicks.

Accepted ONLY if (a) solving with half the frames' lines fixes the line on the OTHER half (held-out check) and
(b) the clicks fit no worse than before. Otherwise the old base stays."""
import json, numpy as np, cv2
from scipy.optimize import least_squares
from . import lines as LN
from .label import pitch_keypoints
from .calcheck import line_mask

def find_near_line(img, camera, pose, L=106.0, W=64.0, band_px=70, min_inliers=200, max_angle_deg=12.0, rng_seed=0):
    """-> {"pts": (15,2) points on the painted near touchline (pixels of this image), "x_range": [xa, xb] metres of the
    model line in view, "inliers": n} or None. pose f is for 1280 wide; img may be any size."""
    h, w = img.shape[:2]; xs = np.arange(0, L + 1e-9, 0.5); P = np.c_[xs, np.full(len(xs), W)]
    q = LN.project(camera, pose, P, w, h)
    ok = np.isfinite(q).all(1) & (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    ok &= np.linalg.norm(P - np.asarray(camera["C"][:2]), axis=1) >= 6.0
    if ok.sum() < 20 or np.linalg.norm(q[ok][0] - q[ok][-1]) < 0.12 * w: return None
    band = np.zeros((h, w), np.uint8); cv2.polylines(band, [np.round(q[ok]).astype(np.int32).reshape(-1, 1, 2)], False, 1, int(2 * band_px * w / 1280))
    band[int(h * 0.84):, int(w * 0.78):] = 0                                     # Veo watermark
    ys, xs_ = np.nonzero(line_mask(img) & band)
    if len(xs_) < min_inliers: return None
    Q = np.c_[xs_, ys].astype(float); rng = np.random.RandomState(rng_seed); best = (0, None)
    d_model = q[ok][-1] - q[ok][0]; d_model /= np.linalg.norm(d_model)
    for _ in range(1500):
        a, b = Q[rng.choice(len(Q), 2, replace=False)]; d = b - a
        if np.linalg.norm(d) < 20: continue
        d /= np.linalg.norm(d)
        if np.degrees(np.arccos(min(1.0, abs(d @ d_model)))) > max_angle_deg: continue
        inl = np.abs((Q - a) @ np.array([-d[1], d[0]])) < 2.5 * w / 1280
        if inl.sum() > best[0]: best = (int(inl.sum()), inl)
    if best[0] < min_inliers: return None
    I = Q[best[1]]; vx, vy, x0, y0 = cv2.fitLine(I.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    d = np.array([vx, vy]); t = (I - [x0, y0]) @ d; ts = np.linspace(np.percentile(t, 2), np.percentile(t, 98), 15)
    if ts[-1] - ts[0] < 0.1 * w: return None                                  # too short to trust
    return {"pts": np.c_[x0 + ts * d[0], y0 + ts * d[1]], "x_range": [float(xs[ok][0]), float(xs[ok][-1])], "inliers": best[0]}

def _line_resid(camera, pose, ev, w, h, W=64.0):
    a, b = LN.project(camera, pose, np.array([[ev["x_range"][0], W], [ev["x_range"][1], W]]), w, h)
    if not (np.isfinite(a).all() and np.isfinite(b).all()): return np.full(len(ev["pts"]), 200.0)
    n = np.array([b[1] - a[1], a[0] - b[0]]); n /= max(np.linalg.norm(n), 1e-9); return (ev["pts"] - a) @ n

def solve(sol, clicks, evidence, w_line=1.0, W=64.0):
    """joint solve: base (C, base tilt) + every frame's pose, from clicks (1280x720) and near-line evidence {frame index: ev}"""
    kp = pitch_keypoints(); obs = []
    for f in sol["frames"]:
        pr = [(kp[n], (u, v)) for n, u, v in clicks[f["session"]][f["frame"]]["pairs"] if "penalty spot" not in n]
        obs.append((np.array([p for p, _ in pr]), np.array([q for _, q in pr], float)))
    cam0 = sol["camera"]; x0 = np.r_[cam0["C"], cam0["base_tilt"], np.array([f["pose"] for f in sol["frames"]], float).ravel()]
    def unpack(x): return {"C": list(x[:3]), "base_tilt": list(x[3:5])}, x[5:].reshape(-1, 4)
    def res(x):
        cam, P = unpack(x); r = [np.nan_to_num(LN.project(cam, p, Pm, 1280, 720) - Q, nan=500).ravel() for (Pm, Q), p in zip(obs, P)]
        r += [w_line * _line_resid(cam, P[i], ev, 1280, 720, W) for i, ev in evidence.items()]
        return np.concatenate(r)
    x = least_squares(res, x0, loss="soft_l1", f_scale=5.0, x_scale="jac", max_nfev=4000).x
    cam, P = unpack(x); out = dict(sol, camera=cam, frames=[dict(f, pose=[float(v) for v in p]) for f, p in zip(sol["frames"], P)])
    return out, obs

def click_errors(sol, obs):
    e = []
    for (Pm, Q), f in zip(obs, sol["frames"]): e += list(np.linalg.norm(LN.project(sol["camera"], f["pose"], Pm, 1280, 720) - Q, axis=1))
    return float(np.median(e)), float(np.percentile(e, 90))

def line_errors(sol, evidence, idx):
    r = [np.abs(_line_resid(sol["camera"], sol["frames"][i]["pose"], evidence[i], 1280, 720)) for i in idx]
    return float(np.median(np.concatenate(r))) if r else None

def fix_base(sol, clicks, images, log=print, min_frames=8, max_click_worse=0.3, min_heldout_gain=0.3):
    """images: {frame index: 1280x720 BGR}. Returns (solution to use, report dict, evidence)."""
    evidence = {}
    for i, img in images.items():
        ev = find_near_line(img, sol["camera"], sol["frames"][i]["pose"])
        if ev: evidence[i] = ev
    idx = sorted(evidence); rep = {"frames_with_near_line": len(idx), "accepted": False}
    base_sol, obs = solve(sol, clicks, {})                                        # clicks only: the fair "before"
    rep["before"] = {"clicks_median_px": round(click_errors(base_sol, obs)[0], 2), "clicks_p90_px": round(click_errors(base_sol, obs)[1], 2),
                     "near_line_median_px": round(line_errors(base_sol, evidence, idx), 1) if idx else None}
    log(f"  painted near touchline found on {len(idx)} of {len(images)} clicked frames")
    if len(idx) < min_frames: rep["why"] = f"only {len(idx)} frames show the near touchline (need {min_frames})"; return sol, rep, evidence
    A, B = idx[0::2], idx[1::2]                                                   # held-out check: solve with A's lines, test on B
    sa, _ = solve(sol, clicks, {i: evidence[i] for i in A})
    rep["heldout"] = {"near_line_B_before_px": round(line_errors(base_sol, evidence, B), 1), "near_line_B_after_px": round(line_errors(sa, evidence, B), 1)}
    new, _ = solve(sol, clicks, evidence); cm, cp = click_errors(new, obs)
    rep["after"] = {"clicks_median_px": round(cm, 2), "clicks_p90_px": round(cp, 2), "near_line_median_px": round(line_errors(new, evidence, idx), 1)}
    rep["base_change"] = {"C_m": [round(a - b, 2) for a, b in zip(new["camera"]["C"], sol["camera"]["C"])],
                          "base_tilt_deg": [round(float(np.degrees(a - b)), 2) for a, b in zip(new["camera"]["base_tilt"], sol["camera"]["base_tilt"])]}
    gain = 1 - rep["heldout"]["near_line_B_after_px"] / max(rep["heldout"]["near_line_B_before_px"], 1e-6)
    ok = gain >= min_heldout_gain and cm <= rep["before"]["clicks_median_px"] + max_click_worse
    rep["accepted"] = bool(ok)
    if not ok: rep["why"] = f"held-out gain {gain:.0%} (need {min_heldout_gain:.0%}) / clicks {cm:.2f} vs {rep['before']['clicks_median_px']:.2f} px"
    log(f"  base fix {'ACCEPTED' if ok else 'REJECTED'}: {json.dumps({k: rep[k] for k in rep if k not in ('accepted',)})}")
    return (new if ok else sol), rep, evidence

def picture(img, old_sol, new_sol, i, ev):
    """old near touchline red, new green, detected painted line yellow dots"""
    out = img.copy(); h, w = img.shape[:2]
    for s, col in ((old_sol, (0, 0, 255)), (new_sol, (0, 255, 0))):
        xs = np.arange(0, 106.01, 0.5); q = LN.project(s["camera"], s["frames"][i]["pose"], np.c_[xs, np.full(len(xs), 64.0)], w, h)
        q = q[np.isfinite(q).all(1) & (np.abs(q) < 5000).all(1)]
        if len(q) > 1: cv2.polylines(out, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], False, col, 2, cv2.LINE_AA)
    for p in ev["pts"]: cv2.circle(out, tuple(np.round(p).astype(int)), 5, (0, 255, 255), 2)
    return out

def transfer_pose(old_cam, new_cam, pose, w=1280, h=720, min_dist_m=25.0, L=106.0, W=64.0):
    """a pose approved under the old base, re-expressed under the new base: same far pitch in the same pixels
    (what Daniel approved), while near lines follow the new base"""
    g = np.array([(x, y) for x in np.arange(0, L + 1e-6, 2.0) for y in np.arange(0, W + 1e-6, 2.0)])
    g = g[np.linalg.norm(g - np.asarray(new_cam["C"][:2]), axis=1) >= min_dist_m]
    q0 = LN.project(old_cam, pose, g, w, h); ok = np.isfinite(q0).all(1) & (q0[:, 0] >= 0) & (q0[:, 0] < w) & (q0[:, 1] >= 0) & (q0[:, 1] < h)
    if ok.sum() < 6: return list(pose)
    g, q0 = g[ok], q0[ok]
    r = least_squares(lambda p: np.nan_to_num(LN.project(new_cam, p, g, w, h) - q0, nan=1e3).ravel(), np.asarray(pose, float), x_scale=[1e-3, 1e-3, 1e-3, 10.0])
    return [float(v) for v in r.x]
