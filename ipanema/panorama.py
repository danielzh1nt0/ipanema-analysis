"""Extend a calibrated panorama so it covers every direction the camera looks (near touchline, both goal ends, far side).

Why: the calibration finds each video frame on a panorama ("the map"). Our first panorama was stitched from one
5-minute stretch that mostly looked at midfield, so views of the near side and the goal ends were not on the map;
those frames inherited the last known position and were badly wrong (15 of the 18 worst frames checked on 21 Sep).

How: start from the verified panorama (its calibration stays valid), pad the canvas, then place frames sampled from
the whole match. A frame is placed only if it registers to the already-covered part of the map with enough, well-spread
matches and a plausible shape, and only if it adds new area. New area is painted fill-only (the verified part is never
altered). Several passes let the map grow outward: frames too far out to match in pass 1 can attach to what pass 1 added.
"""
import cv2, numpy as np
from .mosaic import _feats

def frame_features(sift, frame, scale=0.5):
    """SIFT features of a frame (watermark masked), points in full-resolution frame pixels, descriptors as uint8"""
    kp, d = _feats(sift, frame, scale)
    if d is None or len(kp) < 20: return None
    return np.float32([k.pt for k in kp]) / scale, np.clip(d, 0, 255).astype(np.uint8)

def canvas_features(sift, canvas, covered, scale=0.5):
    small = cv2.resize(canvas, None, fx=scale, fy=scale); m = (cv2.resize(covered.astype(np.uint8), (small.shape[1], small.shape[0])) > 0).astype(np.uint8) * 255
    m = cv2.erode(m, np.ones((5, 5), np.uint8))
    kp, d = sift.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), m)
    if d is None: return None
    return np.float32([k.pt for k in kp]) / scale, np.clip(d, 0, 255).astype(np.uint8)

def _convex(q):
    s = [np.cross(q[(i + 1) % 4] - q[i], q[(i + 2) % 4] - q[(i + 1) % 4]) for i in range(4)]
    return all(v > 0 for v in s) or all(v < 0 for v in s)

def canvas_matcher(cf):
    """search index over the map's features, built once per pass (building it per frame made the real run ~10x slower)"""
    m = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64)); m.add([cf[1].astype(np.float32)]); m.train(); return m

def register(ff, cf, frame_shape, min_inliers=50, min_ratio=0.25, min_spread=0.15, area_range=(0.15, 10.0), matcher=None):
    """frame -> canvas homography, or (None, reason). Guards: inlier count and ratio, spread of inliers across the
    frame, convex footprint of plausible size."""
    if ff is None or cf is None: return None, "no features"
    fpts, fd = ff; cpts, cd = cf
    matcher = matcher or canvas_matcher(cf)
    try: pairs = matcher.knnMatch(fd.astype(np.float32), k=2)
    except cv2.error: return None, "matcher failed"
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < min_inliers: return None, f"{len(good)} matches"
    a = np.float32([fpts[g.queryIdx] for g in good]); b = np.float32([cpts[g.trainIdx] for g in good])
    H, inl = cv2.findHomography(a, b, cv2.RANSAC, 6.0)
    if H is None: return None, "no homography"
    inl = inl.ravel().astype(bool); n = int(inl.sum())
    if n < min_inliers or n < min_ratio * len(good): return None, f"{n} inliers of {len(good)}"
    h, w = frame_shape[:2]; ai = a[inl]
    spread = (np.ptp(ai[:, 0]) * np.ptp(ai[:, 1])) / (w * h)
    if spread < min_spread: return None, f"matches bunched ({spread:.2f} of the frame)"
    q = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), H).reshape(-1, 2)
    if not np.isfinite(q).all() or not _convex(q): return None, "footprint not convex"
    area = cv2.contourArea(q.astype(np.float32)) / (w * h)
    if not (area_range[0] <= area <= area_range[1]): return None, f"footprint {area:.2f}x a frame"
    return H, f"{n} inliers"

def extend(seed, keys, get_frame, pad=(2500, 200, 2500, 1400), max_passes=4, min_gain=0.10, log=print, sift=None):
    """seed: verified panorama (BGR). keys: candidate frame ids; get_frame(key) -> BGR frame.
    Returns canvas, covered mask, offset T (seed px -> canvas px) and the placed frames [(key, H frame->canvas, gain)]."""
    sift = sift or cv2.SIFT_create(nfeatures=4000)
    l, t, r, b = pad; hs, ws = seed.shape[:2]
    canvas = np.zeros((hs + t + b, ws + l + r, 3), np.uint8); canvas[t:t + hs, l:l + ws] = seed
    covered = np.zeros(canvas.shape[:2], bool); covered[t:t + hs, l:l + ws] = seed.max(2) > 8
    T = np.array([[1, 0, l], [0, 1, t], [0, 0, 1.0]])
    feats = {}
    for k in keys:
        f = get_frame(k)
        if f is not None: feats[k] = (frame_features(sift, f), f.shape)
    log(f"panorama: {len(feats)} candidate frames")
    placed, todo = [], [k for k in keys if k in feats]
    csift = cv2.SIFT_create(nfeatures=40000)
    for p in range(max_passes):
        cf = canvas_features(csift, canvas, covered); added = 0; still = []; matcher = canvas_matcher(cf) if cf is not None else None
        for k in todo:
            (ff, shape) = feats[k]; H, why = register(ff, cf, shape, matcher=matcher)
            if H is None: still.append(k); continue
            f = get_frame(k); h, w = shape[:2]
            fm = np.full((h, w), 255, np.uint8); fm[int(h * 0.84):, int(w * 0.78):] = 0          # never paint the watermark
            warped_m = cv2.warpPerspective(fm, H, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_NEAREST) > 0
            new = warped_m & ~covered
            gain = new.sum() / max(1, warped_m.sum())
            if gain < min_gain: continue                                               # already covered: done with it
            warped = cv2.warpPerspective(f, H, (canvas.shape[1], canvas.shape[0]))
            canvas[new] = warped[new]; covered |= new; placed.append((k, H, round(float(gain), 3))); added += 1
        log(f"panorama pass {p + 1}: placed {added}, {len(still)} not yet registrable, coverage {covered.mean() * 100:.1f}% of canvas")
        todo = still
        if added == 0: break
    return canvas, covered, T, placed
