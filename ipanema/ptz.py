"""Track Veo's follow-cam as a pan-tilt-zoom camera inside the panorama (the "ray map").

The panorama camera is fixed and calibrated (cylcam). Every panorama pixel is a known ray from the camera position, so
the panorama is a map of rays: pitch lines, fence, goals, boards, trees, all usable as landmarks. The follow-cam is the
same camera turned and zoomed, so a follow-cam frame is described by pan, tilt, roll and focal length. Matching
follow-cam features to same-moment panorama features gives (pixel, ray) pairs; four numbers are fitted to them with
RANSAC, starting from the previous frame's pose (continuous tracking: no jumping to the other end of the pitch).
Ground-plane homography for the pitch follows directly. (After PTZ-SLAM, Lu et al. 2019, and the two-point method.)"""
import numpy as np, cv2

def rays_from_pano(cam, Q):
    """panorama pixels -> unit ray directions in pitch coordinates (x along, y across, z down)"""
    cx, cy, h, yaw, fu, fv, u0, v0, tilt, roll = cam.params
    Q = np.asarray(Q, float).reshape(-1, 2); az = (Q[:, 0] - u0) / fu + yaw; tb = (Q[:, 1] - v0) / fv
    ct, st, cr, sr = np.cos(tilt), np.sin(tilt), np.cos(roll), np.sin(roll)
    R = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    d = np.column_stack([np.cos(az), np.sin(az), tb]) @ R                   # R^T applied: back to pitch frame
    return d / np.linalg.norm(d, axis=1, keepdims=True)

def rotation(pan, tilt, roll):
    d = np.array([np.cos(tilt) * np.cos(pan), np.cos(tilt) * np.sin(pan), np.sin(tilt)])
    right = np.cross([0, 0, 1.0], d); right /= np.linalg.norm(right)      # z points DOWN: z x forward = image right
    R = np.vstack([right, np.cross(d, right), d])
    c, s = np.cos(roll), np.sin(roll); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R

def project_rays(pose, D, w, h):
    pan, tilt, roll, f = pose; R = rotation(pan, tilt, roll); c = D @ R.T
    z = c[:, 2]; ok = z > 1e-6
    q = np.full((len(D), 2), np.nan); q[ok, 0] = w / 2 + f * c[ok, 0] / z[ok]; q[ok, 1] = h / 2 + f * c[ok, 1] / z[ok]
    return q

def homography(pose, C, w, h):
    """pitch metres -> follow-cam pixels for this pose"""
    pan, tilt, roll, f = pose; R = rotation(pan, tilt, roll); K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    return K @ np.column_stack([R[:, 0], R[:, 1], -R @ np.asarray(C, float)])

def features(img, mask=None, n=5000):
    s = cv2.SIFT_create(nfeatures=n); kp, d = s.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask)
    return (np.float32([k.pt for k in kp]), d) if d is not None and len(kp) >= 20 else (None, None)

def match_pairs(fc, pano, cam, fc_mask=None, pano_mask=None, ratio=0.8):
    """-> follow-cam pixels, rays (from the same-moment panorama frame)"""
    a, da = features(fc, fc_mask); b, db = features(pano, pano_mask)
    if a is None or b is None: return np.zeros((0, 2)), np.zeros((0, 3))
    fl = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64)); pairs = fl.knnMatch(da.astype(np.float32), db.astype(np.float32), k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    P = np.float32([a[g.queryIdx] for g in good]); Q = np.float32([b[g.trainIdx] for g in good])
    return P, rays_from_pano(cam, Q)

def _inliers(pose, P, D, w, h, thr):
    q = np.nan_to_num(project_rays(pose, D, w, h), nan=1e6); return np.linalg.norm(q - P, axis=1) < thr

def fit_pose(P, D, w, h, init=None, thr_px=6.0, seed=0):
    """-> (pose, inlier mask). Start: the previous pose (tracking) or a grid over pan / tilt / zoom scored by inliers
    (no random starts: with few, partly wrong matches a direct search is far more reliable). Then refine on inliers."""
    from scipy.optimize import least_squares
    if len(P) < 6: return None, None
    if init is not None:
        pan0, tilt0, roll0, f0 = init
        cands = [np.array([pan0 + dp, tilt0 + dt, roll0, f0 * df]) for dp in np.radians(np.arange(-12, 13, 2.0)) for dt in np.radians(np.arange(-4, 5, 2.0)) for df in (0.8, 0.9, 1.0, 1.12, 1.25)]
    else:
        cands = [np.array([p, t, 0.0, f]) for p in np.radians(np.arange(-180, 180, 2.0)) for t in np.radians(np.arange(0, 22, 2.0)) for f in np.geomspace(700, 6000, 9)]
    scores = [(int(_inliers(c, P, D, w, h, thr_px * 2.5).sum()), i) for i, c in enumerate(cands)]
    scores.sort(reverse=True)
    best = (0, None, None)
    for n0, i in scores[:6]:
        v = cands[i]
        for thr in (thr_px * 2.5, thr_px):
            inl = _inliers(v, P, D, w, h, thr)
            if inl.sum() < 6: break
            idx = np.where(inl)[0]
            try: v = least_squares(lambda x: np.nan_to_num((project_rays(x, D[idx], w, h) - P[idx]).ravel(), nan=1e3), v, loss="soft_l1", f_scale=thr, max_nfev=300).x
            except Exception: break
        inl = _inliers(v, P, D, w, h, thr_px)
        if inl.sum() > best[0]: best = (int(inl.sum()), v, inl)
    n, v, inl = best
    return (v, inl) if v is not None and n >= 8 else (None, None)

def track(fc_frames, pano_frames, cam, w, h, log=print):
    """pose per frame, continuous: each frame starts from the previous pose; a frame with too few inliers keeps the previous pose"""
    poses, ninl = [], []; prev = None
    fc_mask = np.full((h, w), 255, np.uint8); fc_mask[int(h * 0.84):, int(w * 0.78):] = 0            # Veo watermark
    ph, pw = pano_frames[0].shape[:2]; pano_mask = np.full((ph, pw), 255, np.uint8); pano_mask[:int(0.08 * ph)] = 0; pano_mask[int(0.90 * ph):] = 0
    for i, (fc, pa) in enumerate(zip(fc_frames, pano_frames)):
        P, D = match_pairs(fc, pa, cam, fc_mask, pano_mask)
        v, inl = fit_pose(P, D, w, h, init=prev)
        if v is None and prev is not None: v, inl = fit_pose(P, D, w, h, init=None)         # lost: search again
        if v is None: poses.append(prev); ninl.append(0); continue
        prev = v; poses.append(v); ninl.append(int(inl.sum()))
    return poses, ninl


def track_sequence(frames, C, first_pose, w, h, refine_lines=None, log=print):
    """Carry a pose through consecutive follow-cam frames. Adjacent frames match reliably (same view, small motion):
    features of frame t become rays through pose t, and pose t+1 is fitted to where those features moved (only pan,
    tilt, roll, zoom can change: the base C is fixed). Optional refine_lines(frame, pose) -> pose snaps to painted lines."""
    poses = [np.asarray(first_pose, float)]; ninl = [None]
    fc_mask = np.full((h, w), 255, np.uint8); fc_mask[int(h * 0.84):, int(w * 0.78):] = 0
    prev_pts, prev_desc = features(frames[0], fc_mask)
    for t in range(1, len(frames)):
        pts, desc = features(frames[t], fc_mask); pose = poses[-1]
        if prev_desc is not None and desc is not None:
            fl = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64)); pairs = fl.knnMatch(prev_desc.astype(np.float32), desc.astype(np.float32), k=2)
            good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
            P0 = np.float32([prev_pts[g.queryIdx] for g in good]); P1 = np.float32([pts[g.trainIdx] for g in good])
            if len(P0) >= 12:
                pan, tilt, roll, f = pose; R = rotation(pan, tilt, roll)
                cam = np.column_stack([(P0[:, 0] - w / 2) / f, (P0[:, 1] - h / 2) / f, np.ones(len(P0))]); D = cam @ R      # rays of frame t's features (R^T applied)
                D /= np.linalg.norm(D, axis=1, keepdims=True)
                v, inl = fit_pose(P1, D, w, h, init=pose)
                if v is not None: pose = v; ninl.append(int(inl.sum()))
                else: ninl.append(0)
            else: ninl.append(0)
        else: ninl.append(0)
        if refine_lines is not None: pose = refine_lines(frames[t], pose)
        poses.append(pose); prev_pts, prev_desc = pts, desc
    return poses, ninl
