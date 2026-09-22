"""Curved (cylindrical) camera for Veo's panorama view: a fixed camera on a mast. A pitch point maps to the picture by its
horizontal direction (u) and the tangent of its angle below the horizon (v), after a small tilt/roll of the camera.
Pitch: x along the pitch, y toward the camera side (0 = far touchline). One calibration serves every frame of a clip.

The CylCam object stands in for a per-frame homography everywhere the pipeline converts pixels <-> metres
(calibration.to_m checks for it); `cam @ M` applies a pitch transform first (used to mirror the second half)."""
import numpy as np, cv2

NAMES = ["cx", "cy", "h", "yaw", "fu", "fv", "u0", "v0", "tilt", "roll"]

def _R(tilt, roll):
    ct, st, cr, sr = np.cos(tilt), np.sin(tilt), np.cos(roll), np.sin(roll)
    return np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])

def project(params, P):
    cx, cy, h, yaw, fu, fv, u0, v0, tilt, roll = params
    P = np.asarray(P, float).reshape(-1, 2)
    d = np.column_stack([P[:, 0] - cx, P[:, 1] - cy, np.full(len(P), h)]) @ _R(tilt, roll).T
    az = np.arctan2(d[:, 1], d[:, 0]) - yaw; az = (az + np.pi) % (2 * np.pi) - np.pi
    rho = np.hypot(d[:, 0], d[:, 1]); tb = d[:, 2] / np.maximum(rho, 1e-9)
    return np.column_stack([u0 + fu * az, v0 + fv * tb])

def unproject(params, Q):
    """picture pixels -> pitch metres (ground plane); points at or above the horizon come back as NaN"""
    cx, cy, h, yaw, fu, fv, u0, v0, tilt, roll = params
    Q = np.asarray(Q, float).reshape(-1, 2)
    az = (Q[:, 0] - u0) / fu + yaw; tb = (Q[:, 1] - v0) / fv
    d = np.column_stack([np.cos(az), np.sin(az), tb]) @ _R(tilt, roll)          # rotate back (R orthonormal: R^-1 = R^T)
    s = np.where(d[:, 2] > 1e-9, h / np.maximum(d[:, 2], 1e-9), np.nan)          # ray hits the ground where z = h
    return np.column_stack([cx + s * d[:, 0], cy + s * d[:, 1]])

class CylCam:
    """a curved camera plus an optional pitch transform M applied first (pitch' -> pitch)"""
    def __init__(self, params, M=None):
        self.params = np.asarray(params, float); self.M = np.eye(3) if M is None else np.asarray(M, float)
    def project(self, P):
        P = np.asarray(P, float).reshape(-1, 2); Pm = (np.column_stack([P, np.ones(len(P))]) @ self.M.T)
        return project(self.params, Pm[:, :2] / Pm[:, 2:3])
    def to_m(self, Q):
        m = unproject(self.params, Q); Mi = np.linalg.inv(self.M)
        mh = np.column_stack([m, np.ones(len(m))]) @ Mi.T
        return mh[:, :2] / mh[:, 2:3]
    def __matmul__(self, M): return CylCam(self.params, self.M @ np.asarray(M, float))
    def as_dict(self): return {"camera": "cylindrical", "params": dict(zip(NAMES, map(float, self.params))), "pitch_transform": self.M.tolist()}

def fit(frame, init, L, W, mask_top=0, mask_bottom=None, search=True, log=print):
    """fit the curved camera to the painted lines of a frame, starting from `init` (e.g. another clip's calibration).
    A coarse search over scale and offset first (a recording can show the pitch at a slightly different size or place)."""
    from scipy.optimize import least_squares
    from .calcheck import line_mask, pitch_segments
    h, w = frame.shape[:2]; mb = h if mask_bottom is None else mask_bottom
    m = line_mask(frame); m[:mask_top] = 0; m[mb:] = 0
    dt = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    P = []
    for a, b in pitch_segments(L, W):
        a, b = np.array(a, float), np.array(b, float); n = max(2, int(np.linalg.norm(b - a) / 0.4)); P.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    P = np.vstack(P)
    def resid(p, cap):
        q = project(p, P); ok = np.isfinite(q).all(1) & (q[:, 0] >= 0) & (q[:, 0] < w - 1) & (q[:, 1] >= mask_top) & (q[:, 1] < mb - 1)
        r = np.full(len(q), cap, np.float32)
        if ok.any(): r[ok] = np.minimum(cv2.remap(dt, q[ok, 0].reshape(1, -1).astype(np.float32), q[ok, 1].reshape(1, -1).astype(np.float32), cv2.INTER_LINEAR).ravel(), cap)
        return r
    init = np.asarray(init, float); starts = [init]
    if search:
        for s in (0.85, 0.93, 1.0, 1.08, 1.18):
            for du in (-0.05 * w, 0.0, 0.05 * w):
                for dv in (-0.05 * h, 0.0, 0.05 * h):
                    p = init.copy(); p[4] *= s; p[5] *= s; p[6] = w / 2 + (init[6] - w / 2) * s + du; p[7] = init[7] * s + dv; starts.append(p)
    scored = sorted(starts, key=lambda p: float(np.mean(resid(p, 60.0))))[:4]
    best = None
    for p in scored:
        for cap in (60.0, 25.0, 10.0):
            p = least_squares(lambda v, cap=cap: resid(v, cap), p, loss="soft_l1", f_scale=cap / 4, diff_step=1e-3, max_nfev=500).x
        c = float(np.mean(resid(p, 10.0)))
        if best is None or c < best[0]: best = (c, p)
    q = project(best[1], P); ok = np.isfinite(q).all(1) & (q[:, 0] >= 0) & (q[:, 0] < w - 1) & (q[:, 1] >= mask_top) & (q[:, 1] < mb - 1)
    d = dt[q[ok, 1].astype(int), q[ok, 0].astype(int)]
    stats = {"points": int(ok.sum()), "median_px": round(float(np.median(d)), 1), "p80_px": round(float(np.percentile(d, 80)), 1), "within6_pct": round(float((d <= 6).mean() * 100), 1)}
    log(f"panorama calibration: {stats}")
    return CylCam(best[1]), stats
