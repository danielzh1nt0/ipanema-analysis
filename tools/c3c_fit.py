"""C3c (8 Oct): is it the pitch size or the camera base? Shared-parameter test on raw frames with the near touchline in view.
For each hypothesis (pitch width W, pitch length L, camera height, camera distance behind the near line) every frame's
pose is re-fitted (pan, tilt, roll, zoom; small local search from the row's pose) to the painted white lines, using
  far lines  = every model line point >= 20 m from the camera except the near touchline (what the match fit uses), and
  near line  = the near touchline points (what the match fit leaves out).
Reported per hypothesis: far cost (mean capped px distance to paint), near cost, and the near touchline's median px miss.
A hypothesis that brings the near line onto the paint WITHOUT making the far lines worse explains the drift.
    python tools/c3c_fit.py results/qa/c3c/frames_vall [results/qa/c3c/frames_sfk]   -> <dir>/fit.json"""
import os, sys, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import lines as LN
from ipanema.calcheck import line_mask
from c3c_nearline import paint_mask
from scipy.optimize import minimize

CAP = 15.0
CAP_NEAR = 60.0

def model_pts(camera, L, W, step=0.5):
    """(far points (N,2), near touchline points (M,2)) in metres"""
    C = np.asarray(camera["C"], float)[:2]; far, near = [], None
    for k, P in LN.class_points(L, W, step).items():
        if k == 1: near = P; continue
        far.append(P[np.linalg.norm(P - C, axis=1) >= LN.NEAR_DEFAULT])
    return np.vstack(far), near

class Frame:
    def __init__(self, img):
        self.h, self.w = img.shape[:2]; m = line_mask(img)
        self.paint = int(m.sum()); self.dt = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 5)
        n = paint_mask(img); n[: int(0.4 * self.h)] = 0                      # near line: shaded paint reads sky-blue and wide
        self.dt_near = cv2.distanceTransform((1 - n).astype(np.uint8), cv2.DIST_L2, 5)

    def dists(self, camera, pose, P, dt=None):
        q = LN.project(camera, pose, P, self.w, self.h)
        ok = np.isfinite(q).all(1) & (q[:, 0] >= 0) & (q[:, 0] < self.w) & (q[:, 1] >= 0) & (q[:, 1] < self.h)
        q = q[ok].astype(int); return (self.dt if dt is None else dt)[q[:, 1], q[:, 0]]

    def cost(self, camera, pose, far, near, w_near=1.0, need=30):
        df = self.dists(camera, pose, far); dn = self.dists(camera, pose, near, self.dt_near)
        cf = np.minimum(df, CAP).mean() if len(df) >= need else CAP
        cn = np.minimum(dn, CAP_NEAR).mean() / CAP_NEAR * CAP if len(dn) >= need else CAP   # wider cap, same scale
        return cf + w_near * cn, cf, cn, (float(np.median(np.minimum(dn, 200))) if len(dn) >= need else None)

def refit(fr, camera, pose, far, near, w_near):
    v0 = np.array([pose[0], pose[1], pose[2], np.log(pose[3])])
    obj = lambda v: fr.cost(camera, [v[0], v[1], v[2], np.exp(v[3])], far, near, w_near)[0]
    simplex = np.array([v0, v0 + [0.006, 0, 0, 0], v0 + [0, 0.004, 0, 0], v0 + [0, 0, 0.004, 0], v0 + [0, 0, 0, 0.02]])
    r = minimize(obj, v0, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-3, "maxiter": 400, "initial_simplex": simplex})
    v = r.x; return [v[0], v[1], v[2], np.exp(v[3])]

def camera_variant(camera, dz=0.0, dy=0.0):
    c = {"C": list(np.asarray(camera["C"], float) + [0.0, dy, dz]), "base_tilt": list(camera["base_tilt"])}; return c

def evaluate(frames, poses, camera, L=106.0, W=64.0, dz=0.0, dy=0.0, w_near=1.0, refit_pose=True):
    cam = camera_variant(camera, dz, dy); far, near = model_pts(cam, L, W); out = []
    for fr, p in zip(frames, poses):
        q = refit(fr, cam, p, far, near, w_near) if refit_pose else p
        _, cf, cn, miss = fr.cost(cam, q, far, near); out.append((cf, cn, miss, q))
    cf = np.array([o[0] for o in out]); cn = np.array([o[1] for o in out]); miss = [o[2] for o in out if o[2] is not None]
    return {"L": L, "W": W, "dz": dz, "dy": dy, "w_near": w_near, "refit": refit_pose, "far": round(float(cf.mean()), 2), "near": round(float(cn.mean()), 2),
            "near_miss_px_med": round(float(np.median(miss)), 1) if miss else None, "poses": [list(map(float, o[3])) for o in out],
            "per_frame": [[round(float(o[0]), 2), round(float(o[1]), 2), o[2]] for o in out]}

def load(d, ids=None):
    S = json.load(open(f"{d}/sample.json")); fr, po, ii = [], [], []
    for q in S["rows"]:
        p = f"{d}/{q['id']}.jpg"
        if not os.path.exists(p) or (ids and q["id"] not in ids): continue
        f = Frame(cv2.resize(cv2.imread(p), (1280, 720)))
        if f.paint < 300: continue
        fr.append(f); po.append(np.array(q["pose"], float)); ii.append(q["id"])
    return S, fr, po, ii

def main(argv=sys.argv[1:]):
    for d in argv:
        S, fr, po, ids = load(d); cam = S["camera"]; res = {"folder": d, "frames": ids, "runs": []}
        print(d, len(fr), "frames", flush=True)
        def run(**k):
            r = evaluate(fr, po, cam, **k); res["runs"].append(r)
            print({x: r[x] for x in ("L", "W", "dz", "dy", "w_near", "refit", "far", "near", "near_miss_px_med")}, flush=True); return r
        run(refit_pose=False)                                                   # the rows as they are
        run(w_near=0.0)                                                         # re-fit on far lines only (like the match fit)
        for W in (60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0): run(W=W)           # pitch width, all lines
        for dz in (-1.0, -0.5, 0.5, 1.0): run(dz=dz)                            # camera height (C z is negative = up: -1 = 1 m higher)
        for dy in (-2.0, -1.0, 1.0, 2.0): run(dy=dy)                            # camera distance behind the near line
        for L in (100.0, 103.0, 109.0): run(L=L)
        json.dump(res, open(f"{d}/fit.json", "w"))

if __name__ == "__main__": main()
