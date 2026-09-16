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


def calibrate_via_mosaic(video, clip_id, root, code_dir="/content/ipanema-analysis", log=print):
    """per-frame pitch homography from a hand-calibrated panorama: H_pitch->frame = inv(H_frame->mosaic) @ H_pitch->mosaic"""
    import json, glob
    cal = os.path.join(code_dir, "calibration", f"{clip_id.split('_seg')[0]}.json")
    if not os.path.exists(cal): return None
    spec = json.load(open(cal)); Href = np.array(spec["H_pitch_to_mosaic"], np.float64)
    cache = os.path.join(root, "cache", f"{clip_id}_mosaic_seg.pkl")
    mos = build(video, cache, stride=25, canvas=(4200, 1500), log=log)
    # this segment's panorama is not the calibrated one: register the two panoramas so the calibration transfers
    ref_img = cv2.imread(os.path.join(code_dir, spec["mosaic"]))
    if ref_img is None: log("calibration: reference panorama image missing"); return None
    sift0 = cv2.SIFT_create(nfeatures=8000); bf0 = cv2.BFMatcher(cv2.NORM_L2)
    r = _homog(bf0, *_feats(sift0, mos["mosaic"], 1.0), *_feats(sift0, ref_img, 1.0), scale=1.0, min_inl=40)
    if r is None: log("calibration: could not register this segment's panorama to the calibrated one"); return None
    H_seg_to_ref, n_inl = r; log(f"calibration: panoramas registered ({n_inl} inliers)")
    Hpm = np.linalg.inv(H_seg_to_ref) @ Href
    Hs = {}
    sift = cv2.SIFT_create(nfeatures=3000); bf = cv2.BFMatcher(cv2.NORM_L2)
    keys = sorted(mos["H_to_mosaic"]); ref_feats = {}
    cap = cv2.VideoCapture(video)
    for k in keys:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
        if ok: ref_feats[k] = _feats(sift, f)
    cap.release()
    ok_n = 0
    for k, f in frames(video):
        near = min(keys, key=lambda q: abs(q - k))
        base = mos["H_to_mosaic"][near]
        if k == near: Hfm = base
        else:
            r = _homog(bf, *_feats(sift, f), *ref_feats[near])
            if r is None: continue
            Hfm = base @ r[0]
        try: Hs[k] = np.linalg.inv(Hfm) @ Hpm; ok_n += 1
        except np.linalg.LinAlgError: pass
        if k % 500 == 0: log(f"  mosaic calibration frame {k}")
    log(f"calibration via mosaic: {ok_n} frames from the panorama of {clip_id}")
    return Hs
