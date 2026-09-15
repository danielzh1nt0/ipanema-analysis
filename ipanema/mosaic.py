"""Rebuild the static camera view from a follow-cam clip: register sampled frames to a reference frame and stitch a mosaic.
The mosaic is calibrated once (by hand); each frame's pitch homography = H_mosaic @ H_frame->mosaic."""
import os, pickle, cv2, numpy as np
from .video import frames, info

def _feats(sift, img, scale=0.5):
    g = cv2.cvtColor(cv2.resize(img, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(g, None)

def _homog(bf, k1, d1, k2, d2, scale=0.5, min_inl=25):
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20: return None
    good = [m for m, n in bf.knnMatch(d1, d2, k=2) if m.distance < 0.72 * n.distance]
    if len(good) < min_inl: return None
    p1 = np.float32([k1[m.queryIdx].pt for m in good]); p2 = np.float32([k2[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    if H is None or inl.sum() < min_inl: return None
    S = np.diag([scale, scale, 1.0]); return np.linalg.inv(S) @ H @ S, int(inl.sum())

def build(video, cache, stride=25, canvas=(4200, 1500), log=print):
    """stitch a mosaic; returns {mosaic, H_to_mosaic: {frame_index: 3x3}, ref_index}"""
    if os.path.exists(cache):
        m = pickle.load(open(cache, "rb")); log(f"mosaic: cached ({len(m['H_to_mosaic'])} registered frames)"); return m
    vi = info(video); n = vi["n"]; sift = cv2.SIFT_create(nfeatures=4000); bf = cv2.BFMatcher(cv2.NORM_L2)
    keys = list(range(0, n, stride)); F = {}
    cap = cv2.VideoCapture(video); k = 0; imgs = {}
    while True:
        ok, f = cap.read()
        if not ok: break
        if k in keys: imgs[k] = f
        k += 1
    cap.release(); log(f"mosaic: {len(imgs)} sampled frames")
    for i in keys:
        if i in imgs: F[i] = _feats(sift, imgs[i])
    ref = keys[len(keys) // 2]
    W, Hc = canvas; off = np.array([[1, 0, (W - vi["width"]) / 2], [0, 1, (Hc - vi["height"]) / 2], [0, 0, 1]], np.float64)
    Hs = {ref: off.copy()}
    for direction in (1, -1):
        idx = keys[keys.index(ref)::direction]
        for a, b in zip(idx, idx[1:]):
            if a not in Hs or b not in F or a not in F: continue
            r = _homog(bf, F[b][0], F[b][1], F[a][0], F[a][1])
            if r is None: continue
            Hab, _ = r; Hs[b] = Hs[a] @ Hab
    log(f"mosaic: registered {len(Hs)}/{len(keys)} sampled frames")
    acc = np.zeros((Hc, W, 3), np.float32); cnt = np.zeros((Hc, W, 1), np.float32)
    for i, H in sorted(Hs.items()):
        if i not in imgs: continue
        warp = cv2.warpPerspective(imgs[i].astype(np.float32), H, (W, Hc))
        mask = cv2.warpPerspective(np.ones(imgs[i].shape[:2], np.float32), H, (W, Hc))[..., None]
        acc += warp * mask; cnt += mask
    mosaic = (acc / np.maximum(cnt, 1e-3)).astype(np.uint8)
    out = {"mosaic": mosaic, "H_to_mosaic": Hs, "ref_index": ref, "stride": stride, "size": (W, Hc), "video_size": (vi["width"], vi["height"])}
    pickle.dump(out, open(cache, "wb")); return out

def register_all(video, mos, log=print):
    """homography from every frame to the mosaic (interpolating between sampled frames by direct matching)"""
    sift = cv2.SIFT_create(nfeatures=3000); bf = cv2.BFMatcher(cv2.NORM_L2)
    keys = sorted(mos["H_to_mosaic"]); Hs = {}
    prev_key = None; prev_f = None
    for k, f in frames(video):
        near = min(keys, key=lambda q: abs(q - k))
        if k in mos["H_to_mosaic"]: Hs[k] = mos["H_to_mosaic"][k]; prev_key, prev_f = k, _feats(sift, f); continue
        if prev_f is None: prev_key, prev_f = near, None
        cur = _feats(sift, f)
        base = mos["H_to_mosaic"].get(near)
        if base is None: continue
        # match to the nearest sampled frame
        if prev_key != near or prev_f is None:
            cap = cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_FRAMES, near); ok, rf = cap.read(); cap.release()
            if not ok: continue
            prev_f = _feats(sift, rf); prev_key = near
        r = _homog(bf, cur[0], cur[1], prev_f[0], prev_f[1])
        if r is None: continue
        Hs[k] = base @ r[0]
        if k % 500 == 0: log(f"  mosaic register frame {k}")
    return Hs
