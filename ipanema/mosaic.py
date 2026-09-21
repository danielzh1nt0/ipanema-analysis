"""Rebuild the static camera view from a follow-cam clip: register sampled frames to a reference frame and stitch a mosaic.
The mosaic is calibrated once (by hand); each frame's pitch homography = H_mosaic @ H_frame->mosaic."""
import os, pickle, cv2, numpy as np
from .video import frames, info

def _feats(sift, img, scale=0.5):
    g = cv2.cvtColor(cv2.resize(img, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    # ignore the broadcaster watermark (Veo: bottom-right corner). It is fixed to the screen, not the pitch, so matching it
    # says "the camera didn't move" and folds parts of a pan back onto each other in the stitched panorama.
    h, w = g.shape[:2]; mask = np.full((h, w), 255, np.uint8); mask[int(h * 0.84):, int(w * 0.78):] = 0
    return sift.detectAndCompute(g, mask)

def _homog(bf, k1, d1, k2, d2, scale=0.5, min_inl=25):
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20: return None
    good = [m for m, n in bf.knnMatch(d1, d2, k=2) if m.distance < 0.72 * n.distance]
    if len(good) < min_inl: return None
    p1 = np.float32([k1[m.queryIdx].pt for m in good]); p2 = np.float32([k2[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    if H is None or inl.sum() < min_inl: return None
    S = np.diag([scale, scale, 1.0]); return np.linalg.inv(S) @ H @ S, int(inl.sum())

def build(video, cache, stride=150, canvas=(7000, 2200), log=print):
    """stitch a mosaic by streaming the video (constant memory); returns {mosaic, H_to_mosaic, ...}"""
    if os.path.exists(cache):
        m = pickle.load(open(cache, "rb")); log(f"mosaic: cached ({len(m['H_to_mosaic'])} registered frames)"); return m
    vi = info(video); sift = cv2.SIFT_create(nfeatures=4000); bf = cv2.BFMatcher(cv2.NORM_L2)
    W, Hc = canvas; off = np.array([[1, 0, (W - vi["width"]) / 2], [0, 1, (Hc - vi["height"]) / 2], [0, 0, 1]], np.float64)
    acc = np.zeros((Hc, W, 3), np.float32); cnt = np.zeros((Hc, W, 1), np.float32)
    Hs = {}; prev = None; prevH = None; n_used = 0
    for k, f in frames(video):
        if k % stride: continue
        cur = _feats(sift, f)
        if prev is None: H = off.copy()
        else:
            r = _homog(bf, cur[0], cur[1], prev[0], prev[1])
            if r is None: log(f"  mosaic: lost registration at frame {k}"); prev = cur; continue
            H = prevH @ r[0]
        # guard against drift: reject wild transforms
        corners = cv2.perspectiveTransform(np.float32([[[0, 0]], [[vi["width"], 0]], [[vi["width"], vi["height"]]], [[0, vi["height"]]]]), H).reshape(-1, 2)
        if corners.min() < -W or corners.max() > 2 * W: log(f"  mosaic: drift at frame {k}, skipping"); prev = cur; continue
        Hs[k] = H; prev, prevH = cur, H; n_used += 1
        warp = cv2.warpPerspective(f.astype(np.float32), H, (W, Hc))
        mask = cv2.warpPerspective(np.ones(f.shape[:2], np.float32), H, (W, Hc))[..., None]
        acc += warp * mask; cnt += mask
        if k % (stride * 40) == 0: log(f"  mosaic frame {k} ({n_used} stitched)")
    mosaic = (acc / np.maximum(cnt, 1e-3)).astype(np.uint8)
    out = {"mosaic": mosaic, "H_to_mosaic": Hs, "stride": stride, "size": (W, Hc), "video_size": (vi["width"], vi["height"])}
    log(f"mosaic: {n_used} frames stitched over {vi['n']} ({vi['n']/vi['fps']:.0f} s)")
    pickle.dump({k: v for k, v in out.items() if k != "mosaic"} | {"mosaic": mosaic}, open(cache, "wb")); return out

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


CAL_VERSION = "direct-v3"

def calibrate_via_mosaic(video, clip_id, root, code_dir="/content/ipanema-analysis", step=10, log=print):
    """per-frame pitch homography by registering anchor frames DIRECTLY to the hand-calibrated panorama (no chaining drift);
    frames that don't overlap the panorama chain from the nearest direct anchor."""
    import json, re as _re, bisect
    base = _re.sub(r"_(seg|s)\d+$", "", clip_id)
    cal = os.path.join(code_dir, "calibration", f"{base}.json")
    if not os.path.exists(cal): log(f"calibration: no panorama calibration for {base}"); return None
    spec = json.load(open(cal)); Hpm = np.array(spec["H_pitch_to_mosaic"], np.float64)
    ref_img = cv2.imread(os.path.join(code_dir, spec["mosaic"]))
    if ref_img is None: log("calibration: reference panorama missing"); return None
    sift = cv2.SIFT_create(nfeatures=6000); bf = cv2.BFMatcher(cv2.NORM_L2)
    ref = _feats(sift, ref_img, 1.0)
    anchors = {}; direct = 0; chained = 0; prev = None; prevH = None
    for k, f in frames(video):
        if k % step: continue
        cur = _feats(sift, f, 0.5)
        # direct: frame (0.5 scale feats) -> panorama (1.0 scale feats)
        r = None
        if cur[1] is not None and ref[1] is not None and len(cur[0]) >= 20:
            good = [a for a, b in bf.knnMatch(cur[1], ref[1], k=2) if a.distance < 0.72 * b.distance]
            if len(good) >= 40:
                p1 = np.float32([cur[0][g.queryIdx].pt for g in good]) * 2.0; p2 = np.float32([ref[0][g.trainIdx].pt for g in good])
                Hd, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 4.0)
                if Hd is not None and inl.sum() >= 40: r = (Hd, int(inl.sum()))
        if r is not None: Hfm = r[0]; direct += 1
        elif prev is not None:
            rc = _homog(bf, cur[0], cur[1], prev[0], prev[1])
            if rc is None: prev = cur; continue
            Hfm = prevH @ rc[0]; chained += 1
        else: prev = cur; continue
        anchors[k] = Hfm; prev, prevH = cur, Hfm
        if k % 1000 == 0: log(f"  calibration anchor {k} (direct {direct}, chained {chained})")
    if len(anchors) < 10: log("calibration: too few anchors"); return None
    ak = sorted(anchors); Hs = {}; nfr = 0
    vi = info(video)
    for k in range(vi["n"]):
        i = bisect.bisect_left(ak, k)
        if i < len(ak) and ak[i] == k: Hfm = anchors[k]
        elif i == 0: Hfm = anchors[ak[0]]
        elif i >= len(ak): Hfm = anchors[ak[-1]]
        else:
            a, b = ak[i - 1], ak[i]; t = (k - a) / (b - a)
            Ha, Hb = anchors[a] / anchors[a][2, 2], anchors[b] / anchors[b][2, 2]; Hfm = (1 - t) * Ha + t * Hb
        try: Hs[k] = np.linalg.inv(Hfm) @ Hpm; nfr += 1
        except np.linalg.LinAlgError: pass
    log(f"calibration via panorama: {direct} direct + {chained} chained anchors -> {nfr}/{vi['n']} frames")
    return Hs
