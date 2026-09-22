"""Per-frame pitch->image homography. Keypoint model when the gate passes; feature tracking carries it forward otherwise."""
import sys, pickle, os, cv2, numpy as np
from .video import frames

def _pitch_config(sports_dir):
    sys.path.append(sports_dir)
    from sports.configs.soccer import SoccerPitchConfiguration
    cfg = SoccerPitchConfiguration()
    return np.array(cfg.vertices, dtype=np.float32) / 100.0, cfg.length / 100.0, cfg.width / 100.0

def calibrate(video, weights_pitch, sports_dir, kp_conf=0.5, cache=None, log=print):
    if cache and os.path.exists(cache):
        c = pickle.load(open(cache, "rb")); log(f"calibration: cached ({c['coverage']:.0%} keypoint frames)"); return c
    import supervision as sv
    from ultralytics import YOLO
    VERTS, L, W = _pitch_config(sports_dir)
    model = YOLO(weights_pitch)
    sift = cv2.SIFT_create(nfeatures=3000); bf = cv2.BFMatcher(cv2.NORM_L2); SC = 0.5
    def feats(fr):
        g = cv2.cvtColor(cv2.resize(fr, None, fx=SC, fy=SC), cv2.COLOR_BGR2GRAY); return sift.detectAndCompute(g, None)
    def img_homog(k1, d1, k2, d2):
        if d1 is None or d2 is None or len(k1) < 15 or len(k2) < 15: return None
        good = [m for m, n_ in bf.knnMatch(d1, d2, k=2) if m.distance < 0.75 * n_.distance]
        if len(good) < 15: return None
        p1 = np.float32([k1[m.queryIdx].pt for m in good]); p2 = np.float32([k2[m.trainIdx].pt for m in good])
        Hm, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
        if Hm is None or inl.sum() < 15: return None
        Sm = np.diag([SC, SC, 1.0]); return np.linalg.inv(Sm) @ Hm @ Sm
    H = {}; prev = None; prev_frame = None; prev_feats = None; rejected = 0; carried = 0; n = 0
    for i, f in frames(video):
        n = i + 1
        kp = sv.KeyPoints.from_ultralytics(model(f, verbose=False)[0]); Hk = None
        if len(kp.xy):
            m = kp.confidence[0] > kp_conf
            if m.sum() >= 6:
                Hk, inl = cv2.findHomography(VERTS[m], kp.xy[0][m], cv2.RANSAC, 8.0)
                if Hk is not None:
                    proj = cv2.perspectiveTransform(VERTS[m].reshape(-1, 1, 2), Hk).reshape(-1, 2)
                    err = np.linalg.norm(proj - kp.xy[0][m], axis=1)[inl.ravel() == 1].mean()
                    if inl.sum() < 6 or err > 12: Hk = None
        cur = None
        if Hk is None:
            rejected += 1
            if prev is not None and prev_frame is not None:
                if prev_feats is None: prev_feats = feats(prev_frame)
                cur = feats(f); G = img_homog(prev_feats[0], prev_feats[1], cur[0], cur[1])
                Hk = (G @ prev) if G is not None else prev; carried += G is not None
        elif prev is not None:
            Hk = 0.7 * Hk / Hk[2, 2] + 0.3 * prev / prev[2, 2]
        H[i] = Hk; prev = Hk; prev_frame = f; prev_feats = cur
        if i % 500 == 0: log(f"  calibration frame {i}")
    valid = [k for k in H if H[k] is not None]
    for k in H:
        if H[k] is None: H[k] = H[min(valid, key=lambda v: abs(v - k))]
    c = {"H": H, "L": L, "W": W, "n": n, "coverage": 1 - rejected / max(1, n), "carried": carried, "frozen": rejected - carried}
    log(f"calibration: {n} frames, keypoints on {c['coverage']:.0%}, {carried} carried by feature tracking, {c['frozen']} frozen")
    if cache: pickle.dump(c, open(cache, "wb"))
    return c

PITCH_SEGS_CACHE = {}
def pitch_segments(L, W):
    key = (L, W)
    if key in PITCH_SEGS_CACHE: return PITCH_SEGS_CACHE[key]
    def rect(x0, y0, x1, y1): return [[(x0, y0), (x1, y0)], [(x1, y0), (x1, y1)], [(x1, y1), (x0, y1)], [(x0, y1), (x0, y0)]]
    pb, sb = (W - 40.32) / 2, (W - 18.32) / 2
    segs = rect(0, 0, L, W) + [[(L / 2, 0), (L / 2, W)]] + rect(0, pb, 16.5, W - pb) + rect(L - 16.5, pb, L, W - pb) + rect(0, sb, 5.5, W - sb) + rect(L - 5.5, sb, L, W - sb)
    th = np.linspace(0, 2 * np.pi, 65); circ = [(L / 2 + 9.15 * np.cos(t), W / 2 + 9.15 * np.sin(t)) for t in th]
    segs += [[circ[k], circ[k + 1]] for k in range(64)]
    PITCH_SEGS_CACHE[key] = segs; return segs

def draw_model(frame, Hm, L, W, colour=(0, 0, 255)):
    if hasattr(Hm, "project"):                                 # curved panorama camera: draw each line as a curve
        for a, b in pitch_segments(L, W):
            q = Hm.project(np.array(a, float) + (np.array(b, float) - np.array(a, float)) * np.linspace(0, 1, 60)[:, None])
            q = q[np.isfinite(q).all(1)]
            if len(q) > 1: cv2.polylines(frame, [q.astype(np.int32).reshape(-1, 1, 2)], False, colour, 2)
        return
    for a, b in pitch_segments(L, W):
        pa = Hm @ np.array([a[0], a[1], 1.0]); pb_ = Hm @ np.array([b[0], b[1], 1.0])
        if pa[2] <= 1e-6 or pb_[2] <= 1e-6: continue
        pa, pb_ = pa[:2] / pa[2], pb_[:2] / pb_[2]
        if np.abs(np.r_[pa, pb_]).max() > 20000: continue
        cv2.line(frame, tuple(pa.astype(int)), tuple(pb_.astype(int)), colour, 2)

def to_m(Hm, pts):
    if hasattr(Hm, "to_m"): return Hm.to_m(np.asarray(pts, float).reshape(-1, 2)).astype(np.float32)   # curved panorama camera
    pts = np.float32(pts).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.linalg.inv(Hm).astype(np.float32)).reshape(-1, 2)
