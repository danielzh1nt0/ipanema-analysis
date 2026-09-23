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


def draw_model(img, H, L, W, colour, thick=2):
    for a, b in pitch_segments(L, W):
        p = cv2.perspectiveTransform(np.float32([[a], [b]]), np.float32(H)).reshape(-1, 2)
        if np.isfinite(p).all() and np.abs(p).max() < 1e5: cv2.line(img, tuple(p[0].astype(int)), tuple(p[1].astype(int)), colour, thick)
    return img

def snap_preview(video, H, L, W, to_model=None, every=150, n_worst=3, n_median=1):
    """worst-fitting (and typical) frames of a piece: current calibration in yellow, snapped in red, with fit numbers.
    to_model: 3x3 matrix mapping (L, W) pitch coordinates to the coordinates H was made in (e.g. 106x64 -> 120x70)."""
    T = np.eye(3) if to_model is None else np.asarray(to_model, float); pts = model_points(L, W)
    cap = cv2.VideoCapture(video); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); scored = []
    for k in range(0, n, every):
        if k not in H: continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
        if not ok: continue
        s = score_frame(f, np.asarray(H[k], float) @ T, L, W, pts)
        if s: scored.append((s["p80_px"], k))
    if not scored: cap.release(); return []
    scored.sort(); picks = [k for _, k in scored[-n_worst:]] + [scored[len(scored) // 2][1]] * min(1, n_median)
    out = []
    for kind, k in [("worst", k) for k in picks[:n_worst]] + [("typical", k) for k in picks[n_worst:]]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
        if not ok: continue
        H0 = np.asarray(H[k], float) @ T; Hs, info = refine_to_lines(f, H0, L, W, pts)
        img = draw_model(f.copy(), H0, L, W, (0, 220, 255), 3); img = draw_model(img, Hs, L, W, (0, 0, 255), 2)
        txt = f"{kind} frame {k}: " + (f"SNAPPED p80 {info['p80_before']} -> {info['p80_after']} px, moved {info['shift_px']} px" if info.get("snapped") else f"kept ({info.get('why')})")
        cv2.putText(img, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5); cv2.putText(img, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        out.append({"k": k, "kind": kind, "info": info, "jpg": cv2.imencode(".jpg", cv2.resize(img, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()})
    cap.release(); return out


def confidence_mask(video, H, n, L, W, every=10, bad_px=40.0):
    """'Don't guess': check the calibration against the painted lines every `every` frames. A frame is trusted unless a
    neighbouring check shows the drawn lines more than bad_px from the paint (good frames score ~10-26 px; the misplaced
    ones we inspected scored 58-147 px). Frames with no calibration at all are not trusted. Checks with too few lines in
    view can't judge and don't reject. Returns (ok[n] bool array, summary)."""
    if n and any(hasattr(v, "to_m") for v in list(H.values())[:1]):          # fixed, verified panorama camera
        return np.ones(n, bool), {"checks": 0, "good": 0, "bad": 0, "unjudged": 0, "trusted_pct": 100.0, "static_camera": True}
    pts = model_points(L, W); status = {}
    cap = cv2.VideoCapture(video); k = 0
    while k < n:
        ok_read, f = cap.read()
        if not ok_read: break
        if k % every == 0 and k in H:
            v = judge_frame(f, H[k], L, W)["verdict"]              # the validated check (the old p80 score was fooled by lines in the trees)
            status[k] = None if v == "unjudged" else v
        k += 1
    cap.release()
    ok = np.array([i in H for i in range(n)], bool)
    samples = sorted(status)
    for idx, s in enumerate(samples):
        if status[s] != "bad": continue
        lo = samples[idx - 1] + 1 if idx > 0 else 0; hi = samples[idx + 1] if idx + 1 < len(samples) else n
        ok[lo:hi] = False                                     # everything between the neighbouring checks is untrusted
    vals = list(status.values())
    return ok, {"checks": len(vals), "good": vals.count("good"), "bad": vals.count("bad"), "unjudged": vals.count(None),
                "trusted_pct": round(100 * float(ok.mean()), 1) if n else 0.0}


def judge_frame(frame, H, L, W, min_seg_points=30, weak_support=0.40, max_off_grass=0.03):
    """(size-aware wrapper) The judge was validated on 1920-wide frames: any other size is scaled to 1920 wide first,
    calibration scaled to match, so a verdict means the same at every size (at half size it had passed 4 wrong frames)."""
    h0, w0 = frame.shape[:2]
    if w0 != 1920:
        s = 1920.0 / w0; frame = cv2.resize(frame, (1920, int(round(h0 * s))), interpolation=cv2.INTER_LINEAR)
        H = np.diag([s, s, 1.0]) @ np.asarray(H, float)
        if w0 <= 1100 and weak_support == 0.40:
            # half-size frames blur thin lines: measured on the 25 labelled frames (23 Sep), right frames score 0.30-0.47
            # and wrong ones <= 0.25 (one exception caught by lines off the grass) -> 0.27. Thin margin: keep testing.
            weak_support = 0.27
    return _judge_frame_1920(frame, H, L, W, min_seg_points, weak_support, max_off_grass)

def _judge_frame_1920(frame, H, L, W, min_seg_points=30, weak_support=0.40, max_off_grass=0.03):
    """Is this frame's calibration right? Validated on 28 labelled SFK-BP frames (22 wrong, 3 right, 3 unsure; 21 Sep):
    WRONG if any pitch line clearly in view on the grass has < 40% support from painted lines, or > 3% of the drawn lines
    land off the grass (trees, sky, fence). The old single-number score passed most of the wrong frames."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = cv2.morphologyEx(((hsv[..., 0] > 30) & (hsv[..., 0] < 95) & (hsv[..., 1] > 50)).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8)) > 0
    dt = cv2.distanceTransform((1 - line_mask(frame)).astype(np.uint8), cv2.DIST_L2, 5)
    inview = offg = 0; support = []
    for a, b in pitch_segments(L, W):
        a, b = np.array(a, float), np.array(b, float); m = max(2, int(np.linalg.norm(b - a) / 0.5))
        q = cv2.perspectiveTransform((a + (b - a) * np.linspace(0, 1, m)[:, None]).astype(np.float32).reshape(-1, 1, 2), np.float32(H)).reshape(-1, 2)
        ok = np.isfinite(q).all(1) & (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h); q = q[ok].astype(int)
        if not len(q): continue
        on = grass[q[:, 1], q[:, 0]]; inview += len(q); offg += int((~on).sum())
        if on.sum() >= min_seg_points: support.append(float((dt[q[:, 1], q[:, 0]] <= 8)[on].mean()))
    off = offg / max(1, inview)
    if not support: return {"verdict": "unjudged", "off_grass": round(off, 3), "segments": 0}
    weakest = min(support)
    bad = weakest < weak_support or off > max_off_grass
    return {"verdict": "bad" if bad else "good", "off_grass": round(off, 3), "weakest_support": round(weakest, 3), "segments": len(support)}
