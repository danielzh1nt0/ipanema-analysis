"""Calibrate a Veo follow-cam frame with three numbers: pan, tilt and zoom.

The follow-cam is the same fixed camera as the panorama, only turned and zoomed, so its position (from the panorama fit)
is known and fixed. A pitch point then projects through a plain pinhole camera at that position, and only pan, tilt and
zoom vary per frame. Three unknowns are few enough to search for directly, scored against the frame's own painted lines
(with lines drawn off the grass punished, which is what fooled the earlier attempts)."""
import numpy as np, cv2
from .calcheck import line_mask, pitch_segments

def rotation(pan, tilt, roll=0.0):
    d = np.array([np.cos(tilt) * np.cos(pan), np.cos(tilt) * np.sin(pan), np.sin(tilt)])     # z points into the ground
    right = np.cross([0, 0, 1.0], d); right /= np.linalg.norm(right)      # z points DOWN: z x forward = image right
    R = np.vstack([right, np.cross(d, right), d])
    if roll:
        c, s = np.cos(roll), np.sin(roll); R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R
    return R

def homography(C, pan, tilt, f, roll=0.0, cx=960.0, cy=540.0):
    """pitch metres -> follow-cam pixels"""
    R = rotation(pan, tilt, roll); K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    return K @ np.column_stack([R[:, 0], R[:, 1], -R @ np.asarray(C, float)])

class Scorer:
    """how well a calibration explains this frame: distance of drawn lines to painted lines, plus a penalty for lines
    drawn where there is no grass (trees, sky, fence)"""
    def __init__(self, frame, L, W, step=0.5, off_grass_weight=60.0):
        self.h, self.w = frame.shape[:2]; self.L, self.W = L, W; self.off_w = off_grass_weight
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        grass = ((hsv[..., 0] > 30) & (hsv[..., 0] < 95) & (hsv[..., 1] > 50)).astype(np.uint8)
        self.grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8)) > 0
        self.dt = cv2.distanceTransform((1 - line_mask(frame)).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        pts = []
        for a, b in pitch_segments(L, W):
            a, b = np.array(a, float), np.array(b, float); n = max(2, int(np.linalg.norm(b - a) / step)); pts.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
        self.pts = np.vstack(pts).astype(np.float32)
    def cost(self, H, cap=30.0, need=80):
        q = cv2.perspectiveTransform(self.pts.reshape(-1, 1, 2), np.float32(H)).reshape(-1, 2)
        ok = np.isfinite(q).all(1) & (q[:, 0] >= 0) & (q[:, 0] < self.w - 1) & (q[:, 1] >= 0) & (q[:, 1] < self.h - 1)
        n = int(ok.sum())
        if n < need: return 1e6 - n                                     # too little of the pitch in view to judge
        qi = q[ok].astype(int); on = self.grass[qi[:, 1], qi[:, 0]]
        d = cv2.remap(self.dt, q[ok, 0].reshape(1, -1).astype(np.float32), q[ok, 1].reshape(1, -1).astype(np.float32), cv2.INTER_LINEAR).ravel()
        return float(np.mean(np.minimum(d, cap)) + self.off_w * (~on).mean())

def fit(frame, C, L, W, pan_range=(-175, -5), tilt_range=(1, 30), f_range=(900, 4500), coarse=(3.0, 2.0, 6), log=None):
    """search pan / tilt / zoom, then refine -> (H, info). Nothing is inherited from other frames."""
    from scipy.optimize import minimize
    sc = Scorer(frame, L, W)
    pans = np.radians(np.arange(pan_range[0], pan_range[1], coarse[0])); tilts = np.radians(np.arange(tilt_range[0], tilt_range[1], coarse[1]))
    fs = np.geomspace(f_range[0], f_range[1], coarse[2])
    best = (1e18, None)
    for f in fs:
        for p in pans:
            for t in tilts:
                c = sc.cost(homography(C, p, t, f, cx=sc.w / 2, cy=sc.h / 2))
                if c < best[0]: best = (c, (p, t, f))
    if best[1] is None: return None, {"why": "no pose scored"}
    p0, t0, f0 = best[1]
    cx, cy = sc.w / 2, sc.h / 2                                            # principal point at the centre of THIS frame, whatever its size
    obj = lambda v: sc.cost(homography(C, v[0], v[1], np.exp(v[2]), v[3], cx=cx, cy=cy))
    r = minimize(obj, [p0, t0, np.log(f0), 0.0], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-3, "maxiter": 1200})
    H = homography(C, r.x[0], r.x[1], np.exp(r.x[2]), r.x[3], cx=cx, cy=cy)
    info = {"pan_deg": round(float(np.degrees(r.x[0])), 1), "tilt_deg": round(float(np.degrees(r.x[1])), 1), "zoom": round(float(np.exp(r.x[2]))), "roll_deg": round(float(np.degrees(r.x[3])), 2), "cost": round(float(r.fun), 2)}
    if log: log(f"follow-cam fit: {info}")
    return H, info


def refine(frame, C, L, W, pose, max_pan_deg=1.5, max_tilt_deg=1.0, max_zoom=0.06):
    """snap a tracked pose to this frame's painted lines with a SMALL local search (can't jump to the other end).
    Accepted only if it scores better and stays within the limits; otherwise the tracked pose is kept."""
    from scipy.optimize import minimize
    sc = Scorer(frame, L, W); cx, cy = sc.w / 2, sc.h / 2
    pose = np.asarray(pose, float)
    if (sc.dt <= 0).mean() < 0.002: return pose                            # no painted lines in view: snapping would only drag the pose away
                                                                            # (measured 23 Sep: 17-44 m drift in 5 s without this, 2-4 m with it)
    def obj(v): return sc.cost(homography(C, v[0], v[1], np.exp(v[2]), v[3], cx=cx, cy=cy))
    v0 = np.array([pose[0], pose[1], np.log(pose[3]), pose[2]]); c0 = obj(v0)
    r = minimize(obj, v0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-3, "maxiter": 300, "initial_simplex": np.array([v0, v0 + [0.004, 0, 0, 0], v0 + [0, 0.003, 0, 0], v0 + [0, 0, 0.02, 0], v0 + [0, 0, 0, 0.002]])})
    v = r.x
    if r.fun < c0 and abs(np.degrees(v[0] - pose[0])) <= max_pan_deg and abs(np.degrees(v[1] - pose[1])) <= max_tilt_deg and abs(np.exp(v[2]) / pose[3] - 1) <= max_zoom:
        return np.array([v[0], v[1], v[3], np.exp(v[2])])
    return pose
