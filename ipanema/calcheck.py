"""Calibration quality from the picture itself: how far the drawn pitch model sits from the painted white lines.
Used per frame and per piece to find calibration drift (a piece scoring clearly worse than a verified piece)."""
import numpy as np, cv2
from .calibration import pitch_segments

def line_mask(frame):
    """painted white lines on grass: bright, unsaturated, thin (top-hat), and inside green surroundings"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 1] < 70) & (hsv[..., 2] > 165)
    thin = cv2.morphologyEx(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.MORPH_TOPHAT, np.ones((9, 9), np.uint8)) > 18
    grass = cv2.dilate(((hsv[..., 0] > 30) & (hsv[..., 0] < 95) & (hsv[..., 1] > 60)).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    return (white & thin & grass).astype(np.uint8)

def model_points(L, W, step_m=0.5):
    pts = []
    for a, b in pitch_segments(L, W):
        a, b = np.array(a, float), np.array(b, float); n = max(2, int(np.linalg.norm(b - a) / step_m))
        pts.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    return np.vstack(pts).astype(np.float32)

def score_frame(frame, H, L, W, pts=None, min_points=40):
    """median / 80th-percentile pixel distance from the projected model lines to the nearest painted line; None if too few lines in view"""
    pts = model_points(L, W) if pts is None else pts
    h, w = frame.shape[:2]
    q = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), np.asarray(H, np.float32)).reshape(-1, 2)
    ok = np.isfinite(q).all(1) & (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    q = q[ok]
    if len(q) < min_points: return None
    mask = line_mask(frame)
    if mask.sum() < 200: return None
    dt = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 5)
    d = dt[q[:, 1].astype(int), q[:, 0].astype(int)]
    return {"median_px": float(np.median(d)), "p80_px": float(np.percentile(d, 80)), "points": int(len(d)), "within8_pct": float((d <= 8).mean() * 100)}

def report_piece(video, H, per, L, W, every=150):
    """per-piece calibration report: line-fit scores over ~60 frames, off-pitch detection rate, players per frame,
    and the representative frame (closest to the piece's median score) with the model drawn on it (JPEG bytes)"""
    pts = model_points(L, W); cap = cv2.VideoCapture(video); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); rows = []; frames = {}
    for k in range(0, n, every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
        if not ok or k not in H: continue
        s = score_frame(f, H[k], L, W, pts)
        if s: rows.append((k, s)); frames[k] = f
    cap.release()
    det = [r for k in per for r in per[k]]
    off = sum(1 for r in det if not (0 <= r[2][0] <= L and 0 <= r[2][1] <= W))
    out = {"frames_scored": len(rows), "off_pitch_pct": round(100 * off / max(1, len(det)), 1),
           "players_per_frame": round(len(det) / max(1, len(per)), 1)}
    if not rows: return out, None
    p80 = np.array([s["p80_px"] for _, s in rows]); w8 = np.array([s["within8_pct"] for _, s in rows])
    out.update({"p80_px_median": round(float(np.median(p80)), 1), "p80_px_worst10pct": round(float(np.percentile(p80, 90)), 1),
                "within8_pct_median": round(float(np.median(w8)), 1)})
    k_rep = min(rows, key=lambda r: abs(r[1]["p80_px"] - np.median(p80)))[0]; img = frames[k_rep].copy()
    for a, b in pitch_segments(L, W):
        p = cv2.perspectiveTransform(np.float32([[a], [b]]), np.float32(H[k_rep])).reshape(-1, 2)
        if np.isfinite(p).all() and np.abs(p).max() < 1e5: cv2.line(img, tuple(p[0].astype(int)), tuple(p[1].astype(int)), (0, 0, 255), 2)
    cv2.putText(img, f"frame {k_rep}: p80 {p80.tolist()[[r[0] for r in rows].index(k_rep)]:.1f} px", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return out, cv2.imencode(".jpg", cv2.resize(img, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def refine_to_lines(frame, H, L, W, pts=None, mask=None, caps=(60.0, 25.0, 10.0), max_shift_px=80.0):
    """Snap a pitch->frame homography onto the painted lines visible in this frame (the SoccerNet-winning idea:
    minimise the distance between projected model lines and detected lines). Returns (H, info). Guarded: the snap is
    only kept if it clearly improves the fit and no projected line point moves more than max_shift_px.
    Parameterised by 4 image control points (pixels) with half-pixel finite-difference steps, and the distance map is
    sampled bilinearly, so the optimiser sees a smooth surface (a raw-matrix, whole-pixel version never moved)."""
    from scipy.optimize import least_squares
    pts = model_points(L, W) if pts is None else pts
    mask = line_mask(frame) if mask is None else mask
    h, w = mask.shape
    if mask.sum() < 400: return H, {"snapped": False, "why": "too few painted lines in view"}
    dt = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    H0 = np.asarray(H, float) / H[2, 2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    P = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), np.float32(np.linalg.inv(H0))).reshape(-1, 2)   # pitch points under the image corners
    def H_of(d): return cv2.getPerspectiveTransform(np.float32(P), np.float32(corners + d.reshape(4, 2)))
    def proj(H_): return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), np.float32(H_)).reshape(-1, 2)
    def sample(q, cap):
        ok = np.isfinite(q).all(1) & (q[:, 0] >= 0) & (q[:, 0] < w - 1) & (q[:, 1] >= 0) & (q[:, 1] < h - 1)
        r = np.full(len(q), cap, np.float32)
        if ok.any():
            v = cv2.remap(dt, q[ok, 0].reshape(1, -1).astype(np.float32), q[ok, 1].reshape(1, -1).astype(np.float32), cv2.INTER_LINEAR).ravel()
            r[ok] = np.minimum(v, cap)
        return r, ok
    def fit_stats(H_):
        q = proj(H_); r, ok = sample(q, 1e9)
        if ok.sum() < 40: return None
        d = r[ok]; return float(np.percentile(d, 80)), float((d <= 8).mean())
    base = fit_stats(H0)
    if base is None: return H, {"snapped": False, "why": "too few model lines in view"}
    d = np.zeros(8)
    for cap in caps:
        try: d = least_squares(lambda v, cap=cap: sample(proj(H_of(v)), cap)[0], d, loss="soft_l1", f_scale=cap / 4, diff_step=0.5, max_nfev=300).x
        except Exception: return H, {"snapped": False, "why": "optimiser failed"}
    Hn = H_of(d); new = fit_stats(Hn)
    if new is None: return H, {"snapped": False, "why": "lost the lines"}
    q0, q1 = proj(H0), proj(Hn); vis = np.isfinite(q0).all(1) & np.isfinite(q1).all(1) & (q0[:, 0] >= 0) & (q0[:, 0] < w) & (q0[:, 1] >= 0) & (q0[:, 1] < h)
    shift = float(np.percentile(np.linalg.norm(q1[vis] - q0[vis], axis=1), 95)) if vis.any() else 1e9
    better = new[0] < 0.7 * base[0] and new[1] > base[1] + 0.05
    if better and shift <= max_shift_px:
        return Hn, {"snapped": True, "p80_before": round(base[0], 1), "p80_after": round(new[0], 1), "shift_px": round(shift, 1)}
    return H, {"snapped": False, "why": "no clear improvement" if not better else f"shift {shift:.0f} px too large", "p80": round(base[0], 1)}
