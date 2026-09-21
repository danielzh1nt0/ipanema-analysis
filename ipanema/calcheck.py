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
