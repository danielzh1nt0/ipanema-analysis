"""Link the follow-cam to the panorama of the same Veo recording.

Both videos show the same moment of the same recording: the follow-cam is a zoomed, moving crop of what the panorama
shows. Matching a follow-cam frame to the same-moment panorama frame gives, for each matched feature, its panorama pixel,
which the verified panorama camera turns into pitch metres. A homography fitted from follow-cam pixels to those metres
(RANSAC: only features on the ground plane agree - pitch lines and grass, not players, fence or trees) is that follow-cam
frame's calibration. The ball seen in the follow-cam then maps to metres, and back into the panorama if needed."""
import numpy as np, cv2

def _sift(img, n=6000, mask=None):
    s = cv2.SIFT_create(nfeatures=n); g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, d = s.detectAndCompute(g, mask)
    return (np.float32([k.pt for k in kp]), d) if d is not None and len(kp) >= 20 else (None, None)

def match(fc, pano, fc_mask=None, pano_mask=None, ratio=0.75):
    """-> (follow-cam points, panorama points) of good matches"""
    a, da = _sift(fc, mask=fc_mask); b, db = _sift(pano, mask=pano_mask)
    if a is None or b is None: return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    fl = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
    pairs = fl.knnMatch(da.astype(np.float32), db.astype(np.float32), k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    return np.float32([a[g.queryIdx] for g in good]), np.float32([b[g.trainIdx] for g in good])

def fc_to_metres(fc, pano, cam, L, W, margin=3.0, ransac_m=0.6, pano_mask=None, fc_mask=None):
    """follow-cam frame -> (H mapping follow-cam pixels to pitch metres, info) using the same-moment panorama frame"""
    p_fc, p_pa = match(fc, pano, fc_mask, pano_mask)
    info = {"matches": int(len(p_fc))}
    if len(p_fc) < 12: return None, {**info, "why": "too few matches"}
    m = cam.to_m(p_pa)
    ok = np.isfinite(m).all(1) & (m[:, 0] > -margin) & (m[:, 0] < L + margin) & (m[:, 1] > -margin) & (m[:, 1] < W + margin)
    if ok.sum() < 12: return None, {**info, "why": "too few matches on the pitch"}
    H, inl = cv2.findHomography(p_fc[ok], m[ok].astype(np.float32), cv2.RANSAC, ransac_m)
    if H is None: return None, {**info, "why": "no homography"}
    n = int(inl.sum()); info.update(on_pitch=int(ok.sum()), inliers=n)
    if n < 12: return None, {**info, "why": f"only {n} ground inliers"}
    q = cv2.perspectiveTransform(p_fc[ok][inl.ravel() > 0].reshape(-1, 1, 2), H).reshape(-1, 2)
    info["residual_m"] = round(float(np.median(np.linalg.norm(q - m[ok][inl.ravel() > 0], axis=1))), 3)
    return H, info

def best_frame(fc, pano_frames, cam, L, W, **kw):
    """which of several nearby panorama frames is the same moment as this follow-cam frame (most ground inliers)"""
    best = (-1, None, None)
    for i, pf in enumerate(pano_frames):
        H, info = fc_to_metres(fc, pf, cam, L, W, **kw)
        if H is not None and info["inliers"] > best[0]: best = (info["inliers"], i, H)
    return best
