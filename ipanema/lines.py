"""Line-based follow-cam calibration (next step 1, 24 Sep).

Instead of regressing 31 pitch points, a network paints every visible pitch line with its CLASS (touchline near/far,
goal line, halfway, boxes, circle, D). Labels come for free from verified poses: the 106 x 64 model is drawn through each
verified pose. Classes are mirror-symmetric (no left/right), so horizontal flips are valid augmentation and the network
never has to guess which end it is looking at: the pose fit decides that from perspective, with the camera base KNOWN.

Per frame: predicted class mask -> search pan/tilt/zoom (base fixed) for the pose whose drawn lines match the painted
classes in both directions (model -> picture and picture -> model), then refine. Poses use the repo convention
[pan, tilt, roll, f] with f for a 1280-wide frame; the principal point is the centre of the actual frame."""
import numpy as np, cv2

L_DEF, W_DEF = 106.0, 64.0
CLASSES = ["background", "near touchline", "far touchline", "goal line", "halfway line", "centre circle",
           "penalty box front", "penalty box sides", "goal area front", "goal area sides", "penalty arc"]
COLOURS = [(0, 0, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 255), (255, 0, 255),
           (0, 165, 255), (0, 255, 0), (255, 255, 0), (128, 128, 255), (180, 105, 255)]

def class_segments(L=L_DEF, W=W_DEF, arc_pieces=24):
    """{class id: list of straight pieces ((x0, y0), (x1, y1)) in metres}. Circles/arcs as short straight pieces."""
    c = W / 2; pa, ga = 20.16, 9.16; S = {k: [] for k in range(1, len(CLASSES))}
    S[1] = [((0, W), (L, W))]; S[2] = [((0, 0), (L, 0))]; S[3] = [((0, 0), (0, W)), ((L, 0), (L, W))]
    S[4] = [((L / 2, 0), (L / 2, W))]
    th = np.linspace(0, 2 * np.pi, 3 * arc_pieces + 1); p = np.c_[L / 2 + 9.15 * np.cos(th), c + 9.15 * np.sin(th)]
    S[5] = [(tuple(p[i]), tuple(p[i + 1])) for i in range(len(p) - 1)]
    for x0, s in ((0.0, 1), (L, -1)):
        S[6].append(((x0 + s * 16.5, c - pa), (x0 + s * 16.5, c + pa)))
        S[7] += [((x0, c - pa), (x0 + s * 16.5, c - pa)), ((x0, c + pa), (x0 + s * 16.5, c + pa))]
        S[8].append(((x0 + s * 5.5, c - ga), (x0 + s * 5.5, c + ga)))
        S[9] += [((x0, c - ga), (x0 + s * 5.5, c - ga)), ((x0, c + ga), (x0 + s * 5.5, c + ga))]
        a = np.arccos(5.5 / 9.15)                                          # the D: part of the r=9.15 circle round the spot outside the box
        th = np.linspace(-a, a, arc_pieces + 1); p = np.c_[x0 + s * (11.0 + 9.15 * np.cos(th)), c + 9.15 * np.sin(th)]
        S[10] += [(tuple(p[i]), tuple(p[i + 1])) for i in range(len(p) - 1)]
    return S

def class_points(L=L_DEF, W=W_DEF, step=0.5):
    """dense model points per class (for the model -> picture direction)"""
    out = {}
    for k, segs in class_segments(L, W).items():
        pts = []
        for a, b in segs:
            a, b = np.array(a, float), np.array(b, float); n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
            pts.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
        out[k] = np.vstack(pts)
    return out

# ---- camera (same model as kptrain.project / fccam.homography, in double precision) ----
def _rot(pan, tilt, roll):
    d = np.array([np.cos(tilt) * np.cos(pan), np.cos(tilt) * np.sin(pan), np.sin(tilt)])
    r = np.cross([0, 0, 1.0], d); r /= np.linalg.norm(r); R = np.vstack([r, np.cross(d, r), d])
    c, s = np.cos(roll), np.sin(roll); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R

def _base(bx, by):
    cx, sx, cy, sy = np.cos(bx), np.sin(bx), np.cos(by), np.sin(by)
    return np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])

def to_cam(camera, pose, P):
    """pitch metres (N,2) -> camera coords (N,3); z > 0 is in front"""
    R = _rot(*pose[:3]) @ _base(*camera["base_tilt"])
    return (np.column_stack([np.asarray(P, float), np.zeros(len(P))]) - np.asarray(camera["C"], float)) @ R.T

def project(camera, pose, P, w, h):
    """pitch metres -> pixels of a w x h frame (NaN behind the camera). pose f is for 1280 wide."""
    c = to_cam(camera, pose, P); f = pose[3] * w / 1280.0
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.column_stack([w / 2 + f * c[:, 0] / c[:, 2], h / 2 + f * c[:, 1] / c[:, 2]])
    q[c[:, 2] <= 1e-3] = np.nan; return q

def projected_segments(camera, pose, w, h, L=L_DEF, W=W_DEF, segs=None, z_near=0.5):
    """{class: (M,4) pixel segments x0,y0,x1,y1}, each piece clipped in 3-D at z_near (in front of the camera) and to a
    generous box around the frame. Straight pitch lines stay straight in a pinhole picture, so this is exact."""
    segs = class_segments(L, W) if segs is None else segs; f = pose[3] * w / 1280.0; out = {}
    for k, ss in segs.items():
        if not len(ss): out[k] = np.zeros((0, 4)); continue
        A = to_cam(camera, pose, [a for a, b in ss]); B = to_cam(camera, pose, [b for a, b in ss]); rows = []
        for a, b in zip(A, B):
            if a[2] < z_near and b[2] < z_near: continue
            if a[2] < z_near: a = a + (b - a) * (z_near - a[2]) / (b[2] - a[2])
            elif b[2] < z_near: b = b + (a - b) * (z_near - b[2]) / (a[2] - b[2])
            pa = np.array([w / 2 + f * a[0] / a[2], h / 2 + f * a[1] / a[2]]); pb = np.array([w / 2 + f * b[0] / b[2], h / 2 + f * b[1] / b[2]])
            ok, p1, p2 = cv2.clipLine((-2 * w, -2 * h, 5 * w, 5 * h), tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int))) \
                if np.abs(np.r_[pa, pb]).max() < 1e9 else (False, None, None)
            if ok: rows.append([*pa, *pb] if np.abs(np.r_[pa, pb]).max() < 1e6 else [*p1, *p2])
        out[k] = np.array(rows, float).reshape(-1, 4)
    return out

IGNORE = 255
NEAR_M = {1: 45.0}     # ground distance to the camera below which a class is "don't know" in labels and left out of the fit.
NEAR_DEFAULT = 20.0    # Measured 24 Sep on fc_0313.5 (clicked pose): the drawn near touchline sits 30-44 px (1280) off the painted
                       # one along its whole visible length, although Daniel's far clicks fit to 4-5 px. A joint re-solve with that
                       # painted line added moves the base tilt 0.3 deg and fixes it without hurting the clicks: the base is loosely
                       # pinned in the near field. Until that is checked on more frames, near-camera labels are not trusted.

def _near_m(k, near_m=None):
    near_m = NEAR_M if near_m is None else near_m
    return near_m.get(k, NEAR_DEFAULT) if isinstance(near_m, dict) else float(near_m)

def _split_near(segs, C, near_m=None, piece=1.0):
    """cut every piece into ~1 m bits and sort them into (far, near) by ground distance of the bit's middle to the camera"""
    far, near = {}, {}
    for k, ss in segs.items():
        lim = _near_m(k, near_m)
        far[k], near[k] = [], []
        for a, b in ss:
            a, b = np.array(a, float), np.array(b, float); n = max(1, int(np.ceil(np.linalg.norm(b - a) / piece)))
            for i in range(n):
                p, q = a + (b - a) * i / n, a + (b - a) * (i + 1) / n
                (near if np.linalg.norm((p + q) / 2 - np.asarray(C[:2], float)) < lim else far)[k].append((tuple(p), tuple(q)))
    return far, near

def render_mask(camera, pose, w, h, thick=None, L=L_DEF, W=W_DEF, near_m=None, band=0.04):
    """uint8 class-id mask (0 = no line, 255 = don't know) of the model drawn through a pose: the training label.
    Lines closer than near_m to the camera are drawn as a wide 'don't know' band instead of a class."""
    thick = max(2, int(round(w / 320))) if thick is None else thick; m = np.zeros((h, w), np.uint8)
    far, near = _split_near(class_segments(L, W), camera["C"], near_m) if near_m != 0 else (class_segments(L, W), {})
    def draw(segs, val_of, t):
        S = projected_segments(camera, pose, w, h, L, W, segs)
        for k in sorted(S, key=lambda k: k in (1, 2, 3)):                                  # boundary lines drawn last (win overlaps)
            for x0, y0, x1, y1 in S[k]:
                ok, p1, p2 = cv2.clipLine((0, 0, w, h), (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))))
                if ok: cv2.line(m, p1, p2, val_of(k), t, cv2.LINE_8)
    draw(far, int, thick)
    if near: draw(near, lambda k: IGNORE, max(thick, int(round(2 * band * w))))      # wide: the painted line is somewhere in there
    return m

def overlay(img, mask, alpha=0.8):
    pal = np.zeros((256, 3), np.uint8); pal[:len(COLOURS)] = COLOURS; pal[IGNORE] = (90, 90, 90)
    out = img.copy(); col = pal[mask]; on = mask > 0
    out[on] = (alpha * col[on] + (1 - alpha) * out[on]).astype(np.uint8); return out

def legend(img):
    for i, n in enumerate(CLASSES[1:], 1):
        cv2.putText(img, n, (8, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3); cv2.putText(img, n, (8, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOURS[i], 1)
    return img

# ---- grading: how far apart are two poses, in pixels, over the pitch actually in view ----
def click_error(camera, pose, pairs, w=1280, h=720):
    """THE grade (25 Sep): pixel distance (1280 frame) between where a pose puts Daniel's clicked points and where he
    clicked them. Penalty spots left out (as in the base solve). Unlike pose_error it needs no reference pose, which
    carries its own 5-15 px uncertainty and differs most on near grass that nobody clicked."""
    from .label import pitch_keypoints
    kp = pitch_keypoints(); pr = [(kp[n], (u, v)) for n, u, v in pairs if "penalty spot" not in n and n in kp]
    if not pr or pose is None: return {"median_px": float("inf"), "max_px": float("inf"), "clicks": 0}
    q = project(camera, pose, np.array([p for p, _ in pr]), w, h); d = np.linalg.norm(np.nan_to_num(q, nan=1e6) - np.array([c for _, c in pr]), axis=1)
    return {"median_px": float(np.median(d)), "max_px": float(d.max()), "clicks": len(pr)}

def pose_error(camera, pose, ref, w=1280, h=720, L=L_DEF, W=W_DEF, step=2.0):
    """median / 90th percentile pixel distance between where `pose` and `ref` put the pitch (grid every 2 m, only points
    in view under ref). This is what matters for positions: a wrong pose that happens to share a line still scores badly."""
    g = np.array([(x, y) for x in np.arange(0, L + 1e-6, step) for y in np.arange(0, W + 1e-6, step)])
    a = project(camera, ref, g, w, h); b = project(camera, pose, g, w, h)
    ok = np.isfinite(a).all(1) & (a[:, 0] >= 0) & (a[:, 0] < w) & (a[:, 1] >= 0) & (a[:, 1] < h)
    if ok.sum() < 3: return {"median_px": np.inf, "p90_px": np.inf, "points": int(ok.sum())}
    d = np.linalg.norm(np.nan_to_num(b[ok], nan=1e6) - a[ok], axis=1)
    return {"median_px": float(np.median(d)), "p90_px": float(np.percentile(d, 90)), "points": int(ok.sum())}

# ---- fitting a pose to a predicted class mask ----
def _pt_seg_dist(P, S):
    """(N,2) points x (M,4) segments -> (N,) distance to the nearest segment"""
    if len(S) == 0 or len(P) == 0: return np.full(len(P), np.inf)
    a = S[None, :, :2]; b = S[None, :, 2:]; ab = b - a; ap = P[:, None, :] - a
    t = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
    return np.sqrt(((ap - t[..., None] * ab) ** 2).sum(-1)).min(1)

def _rot_batch(pan, tilt, roll):
    """(G,) angles -> (G,3,3) rotations, same convention as _rot"""
    d = np.stack([np.cos(tilt) * np.cos(pan), np.cos(tilt) * np.sin(pan), np.sin(tilt)], -1)
    r = np.stack([-d[:, 1], d[:, 0], np.zeros_like(pan)], -1); r /= np.linalg.norm(r, axis=1, keepdims=True)
    R = np.stack([r, np.cross(d, r), d], 1); c, s_ = np.cos(roll), np.sin(roll); Z = np.zeros_like(pan); O = np.ones_like(pan)
    Rr = np.stack([np.stack([c, -s_, Z], -1), np.stack([s_, c, Z], -1), np.stack([Z, Z, O], -1)], 1)
    return Rr @ R

class MaskScorer:
    """How well a pose explains a predicted class mask. Both directions, per class:
    model -> picture: projected model points of class k should land on predicted class-k pixels (distance transform);
    picture -> model: predicted class-k pixels should lie on the projected class-k lines.
    Distances are capped so a few wrong pixels (players, missed lines) can't dominate. Classes are never mixed up:
    a box line can't be explained by the halfway line, which is what fooled the class-blind fits on corner views."""
    def __init__(self, mask, camera, L=L_DEF, W=W_DEF, cap_frac=0.04, n_img=800, seed=0, min_pixels=15, step_m=1.0):
        self.h, self.w = mask.shape; self.cam = camera; self.cap = cap_frac * self.w; self.L, self.W = L, W
        self.B = _base(*camera["base_tilt"]); self.C = np.asarray(camera["C"], float)
        g = self.C[:2]; pts, cls = [], []
        for k, P in class_points(L, W, step=step_m).items():
            P = P[np.linalg.norm(P - g, axis=1) >= _near_m(k)]                  # unreliable near the camera
            pts.append(P); cls.append(np.full(len(P), k))
        self.P3 = np.column_stack([np.vstack(pts), np.zeros(sum(len(p) for p in pts))]) - self.C; self.cls = np.concatenate(cls)
        rng = np.random.RandomState(seed); self.img = {}; self.n_pix = {}; stack = []; self.present = np.zeros(len(CLASSES), bool)
        n_all = max(1, int(((mask > 0) & (mask != IGNORE)).sum()))
        for k in range(1, len(CLASSES)):
            m = (mask == k)
            if m.sum() >= min_pixels:
                self.present[k] = True                                         # picture pixels sampled in proportion to each class's size,
                ys, xs = np.nonzero(m); n = max(5, int(round(n_img * len(xs) / n_all)))   # so a small stray blob can't outvote a real line
                i = rng.choice(len(xs), min(n, len(xs)), replace=False); self.img[k] = np.c_[xs[i], ys[i]].astype(float); self.n_pix[k] = len(xs)
                stack.append(np.minimum(cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5), self.cap).astype(np.float32))
            else: stack.append(np.full((self.h, self.w), 0.6 * self.cap, np.float32))   # class not seen: missed line or wrong pose
        self.dts = np.stack(stack)                                                    # (K-1, h, w), already capped
        self.n_pred = sum(len(v) for v in self.img.values())
        segs = class_segments(L, W); A, Bp, sc = [], [], []
        for k, ss in segs.items():
            for a, b in ss: A.append(a); Bp.append(b); sc.append(k)
        self.SA = np.column_stack([np.array(A, float), np.zeros(len(A))]) - self.C; self.SB = np.column_stack([np.array(Bp, float), np.zeros(len(Bp))]) - self.C
        self.scls = np.array(sc)

    def model_to_image_batch(self, poses, need=25):
        """(G,4) poses -> (G,) mean capped distance of in-view model points to their own class in the picture"""
        poses = np.atleast_2d(np.asarray(poses, float)); R = _rot_batch(poses[:, 0], poses[:, 1], poses[:, 2]) @ self.B
        c = np.einsum("gij,nj->gni", R, self.P3); f = (poses[:, 3] * self.w / 1280.0)[:, None]
        z = c[..., 2]; front = z > 1e-3; zs = np.where(front, z, 1.0)
        u = self.w / 2 + f * c[..., 0] / zs; v = self.h / 2 + f * c[..., 1] / zs
        ok = front & (u >= 0) & (u <= self.w - 1) & (v >= 0) & (v <= self.h - 1)
        ui = np.clip(np.round(u), 0, self.w - 1).astype(np.int32); vi = np.clip(np.round(v), 0, self.h - 1).astype(np.int32)
        d = self.dts[self.cls[None, :] - 1, vi, ui]; n = ok.sum(1)
        out = np.where(ok, d, 0).sum(1) / np.maximum(n, 1)
        return np.where(n >= need, out, 10 * self.cap + (need - n))

    def model_to_image(self, pose, need=25): return float(self.model_to_image_batch([pose], need)[0])

    def segs_px(self, pose, z_near=0.5):
        """(M,4) pixel segments of every model piece, clipped at z_near in front of the camera, and their classes"""
        R = _rot(*pose[:3]) @ self.B; A = self.SA @ R.T; Bc = self.SB @ R.T; f = pose[3] * self.w / 1280.0
        keep = (A[:, 2] >= z_near) | (Bc[:, 2] >= z_near); A, Bc, k = A[keep], Bc[keep], self.scls[keep]
        ta = np.clip((z_near - A[:, 2]) / np.where(np.abs(Bc[:, 2] - A[:, 2]) > 1e-12, Bc[:, 2] - A[:, 2], 1e-12), 0, 1)[:, None]
        A2 = np.where((A[:, 2] < z_near)[:, None], A + (Bc - A) * ta, A)
        tb = np.clip((z_near - Bc[:, 2]) / np.where(np.abs(A[:, 2] - Bc[:, 2]) > 1e-12, A[:, 2] - Bc[:, 2], 1e-12), 0, 1)[:, None]
        B2 = np.where((Bc[:, 2] < z_near)[:, None], Bc + (A - Bc) * tb, Bc)
        return np.column_stack([self.w / 2 + f * A2[:, 0] / A2[:, 2], self.h / 2 + f * A2[:, 1] / A2[:, 2], self.w / 2 + f * B2[:, 0] / B2[:, 2], self.h / 2 + f * B2[:, 1] / B2[:, 2]]), k

    def image_to_model(self, pose, z_near=0.5):
        if not self.n_pred: return self.cap
        R = _rot(*pose[:3]) @ self.B; A = self.SA @ R.T; Bc = self.SB @ R.T; f = pose[3] * self.w / 1280.0
        keep = (A[:, 2] >= z_near) | (Bc[:, 2] >= z_near)
        A, Bc, k = A[keep], Bc[keep], self.scls[keep]
        ta = np.clip((z_near - A[:, 2]) / np.where(np.abs(Bc[:, 2] - A[:, 2]) > 1e-12, Bc[:, 2] - A[:, 2], 1e-12), 0, 1)[:, None]
        A2 = np.where((A[:, 2] < z_near)[:, None], A + (Bc - A) * ta, A)
        tb = np.clip((z_near - Bc[:, 2]) / np.where(np.abs(A[:, 2] - Bc[:, 2]) > 1e-12, A[:, 2] - Bc[:, 2], 1e-12), 0, 1)[:, None]
        B2 = np.where((Bc[:, 2] < z_near)[:, None], Bc + (A - Bc) * tb, Bc)
        S = np.column_stack([self.w / 2 + f * A2[:, 0] / A2[:, 2], self.h / 2 + f * A2[:, 1] / A2[:, 2], self.w / 2 + f * B2[:, 0] / B2[:, 2], self.h / 2 + f * B2[:, 1] / B2[:, 2]])
        tot = 0.0
        for kk, P in self.img.items(): tot += np.minimum(_pt_seg_dist(P, S[k == kk]), self.cap).sum()
        return tot / self.n_pred

    def cost(self, pose): return self.model_to_image(pose) + self.image_to_model(pose)

    FAMILY = {1: "along", 2: "along", 7: "along", 9: "along", 3: "across", 4: "across", 6: "across", 8: "across", 5: "circle", 10: "arc"}
    def support(self, pose, near_frac=0.015, min_px=40, min_share=0.6):
        """line families (along the pitch / across it / circle / D) that the pose explains with real evidence: enough of the
        class's predicted pixels lie on its drawn line. One family alone (e.g. only the far touchline) can't place a frame."""
        SS, kk = self.segs_px(pose); fams = set()
        for k, P in self.img.items():
            S = SS[kk == k]
            if not len(S): continue
            on = _pt_seg_dist(P, S) <= near_frac * self.w; n_on = on.mean() * self.n_pix[k]
            if on.mean() >= min_share and n_on >= min_px: fams.add(self.FAMILY[k])
        return fams

    def unexplained(self, pose, near_frac=0.015, big_px=150, min_share=0.5):
        """classes the network paints clearly (big_px pixels or more) that the pose does NOT explain (under min_share of
        their pixels on the drawn line). One such class means the pose is wrong somewhere, whatever else fits
        (fc_0388 on 25 Sep: far lines fitted, painted centre circle 40 px off, still called confident)."""
        SS, kk = self.segs_px(pose); bad = []
        for k, P in self.img.items():
            if self.n_pix[k] < big_px: continue
            S = SS[kk == k]
            if not len(S): bad.append(CLASSES[k]); continue
            if (_pt_seg_dist(P, S) <= near_frac * self.w).mean() < min_share: bad.append(CLASSES[k])
        return bad

def fit_pose(mask, camera, pan_deg=(-178, -2), tilt_deg=(1, 28), f1280=(250, 4000), steps=(2.0, 1.0, 16), keep=8,
             init=None, L=L_DEF, W=W_DEF, log=None, rival_px=25.0, min_margin=0.35, max_cost_frac=0.02):
    """Pose [pan, tilt, roll, f(1280)] from a predicted class mask, camera base known. Coarse grid on the fast cost,
    then the best `keep` distinct candidates are refined on the two-way cost; the lowest wins. init (a pose) skips the
    grid and only refines around it (for tracking). Returns (pose, info).
    info["confident"] is False when a clearly different view (> rival_px apart) explains the mask almost as well (within
    min_margin) or the fit is poor: then the answer must not be used on its own (the tracker carries the pose instead)."""
    from scipy.optimize import minimize
    sc = MaskScorer(mask, camera, L, W)
    if not sc.img: return None, {"why": "no lines predicted"}
    if init is None:
        F, T, P = np.meshgrid(np.geomspace(*f1280, steps[2]), np.radians(np.arange(tilt_deg[0], tilt_deg[1] + 1e-9, steps[1])),
                              np.radians(np.arange(pan_deg[0], pan_deg[1] + 1e-9, steps[0])), indexing="ij")
        G = np.column_stack([P.ravel(), T.ravel(), np.zeros(P.size), F.ravel()])
        c = np.concatenate([sc.model_to_image_batch(G[i:i + 400]) for i in range(0, len(G), 400)])
        starts = []
        for i in np.argsort(c):                                                # distinct starts: not the same view twice
            p, t, _, f = G[i]
            if all(abs(p - q[0]) > np.radians(6) or abs(np.log(f / q[3])) > 0.35 or abs(t - q[1]) > np.radians(3) for q in starts): starts.append(G[i])
            if len(starts) >= keep: break
    else: starts = [np.asarray(init, float)]
    obj = lambda v: sc.cost([v[0], v[1], v[2], np.exp(v[3])]); sols = []
    for s in starts:
        v0 = np.array([s[0], s[1], s[2], np.log(s[3])])
        simplex = np.array([v0, v0 + [0.02, 0, 0, 0], v0 + [0, 0.01, 0, 0], v0 + [0, 0, 0.01, 0], v0 + [0, 0, 0, 0.08]])
        r = minimize(obj, v0, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-4, "maxiter": 1500, "initial_simplex": simplex})
        sols.append((float(r.fun), r.x))
    sols.sort(key=lambda t: t[0]); best = sols[0]; v = best[1]
    as_pose = lambda u: np.array([u[0], u[1], u[2], np.exp(u[3])])
    rival = next(((c, u) for c, u in sols[1:] if pose_error(camera, as_pose(u), as_pose(v), L=L, W=W)["median_px"] > rival_px), None)
    margin = (rival[0] - best[0]) / max(best[0], 0.5) if rival else np.inf       # how much worse the best DIFFERENT view is
    fine = MaskScorer(mask, camera, L, W, cap_frac=0.015)                    # final polish with a tight cap: exact lines decide
    r = minimize(lambda u: fine.cost([u[0], u[1], u[2], np.exp(u[3])]), v, method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-5, "maxiter": 800, "initial_simplex": np.array([v, v + [0.003, 0, 0, 0], v + [0, 0.002, 0, 0], v + [0, 0, 0.002, 0], v + [0, 0, 0, 0.01]])})
    if r.fun <= fine.cost([v[0], v[1], v[2], np.exp(v[3])]): v = r.x
    pose = np.array([v[0], v[1], v[2], np.exp(v[3])])
    fams = fine.support(pose); bad = fine.unexplained(pose)
    confident = bool(margin >= min_margin and best[0] <= max_cost_frac * sc.w and len(fams) >= 2 and not bad)
    info = {"confident": confident, "margin": round(float(margin), 2) if np.isfinite(margin) else None, "cost": round(float(best[0]), 3), "cost_px_at_width": sc.w, "supported_families": sorted(fams), "unexplained": bad, "classes_seen": [CLASSES[k] for k in sorted(sc.img)],
            "pan_deg": round(float(np.degrees(pose[0])), 2), "tilt_deg": round(float(np.degrees(pose[1])), 2), "f1280": round(float(pose[3]), 1)}
    if log: log(f"line fit: {info}")
    return pose, info


# ---- training data from verified poses ----
def held_back(sol, val_every=6):
    """the SAME held-back clicked frames as the point detector (grading stays comparable): every val_every-th clicked frame"""
    return [f for i, f in enumerate(sol["frames"]) if i % val_every == val_every - 1 and not f.get("source")]

def build_dataset(solution_json, frame_zips, out_dir, random_json=None, random_review=None, random_zip=None,
                  size=(640, 360), val_every=6, keep_away_s=10.0, log=print, pose_fn=None, full_val=True, near_m=None):
    """images/{train,val}/*.jpg + masks/{train,val}/*.png (class ids, 255 = don't know) + poses.json.
    Train = clicked frames not held back + random-moment frames Daniel said YES to (none within keep_away_s of a held-back
    moment). Val = the held-back clicked frames only. Propagated neighbours are NOT used: near-copies add volume, not variety."""
    import os, json, zipfile
    sol = json.load(open(solution_json)) if isinstance(solution_json, str) else solution_json; cam = sol["camera"]; w, h = size
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clicks = {s: json.load(open(f"{root}/results/labels/SFKBP1109_points_{s}.json")) for s in ("s1", "s2") if os.path.exists(f"{root}/results/labels/SFKBP1109_points_{s}.json")}
    val = {(f["session"], f["frame"]) for f in held_back(sol, val_every)}; val_t = [float(n[3:-4]) for _, n in val]
    items = [(f["session"], f["frame"], f["pose"], "val" if (f["session"], f["frame"]) in val else "train") for f in sol["frames"]]
    if random_json and random_review and random_zip and os.path.exists(random_review):
        prop = json.load(open(random_json)); rev = json.load(open(random_review)); frame_zips = dict(frame_zips, random=random_zip); n_near = 0
        for name, v in sorted(prop.items()):
            if rev.get(name) != "yes": continue
            if min(abs(float(name[3:-4]) - t) for t in val_t) < keep_away_s: n_near += 1; continue
            p = list(v["pose"]); p[3] *= 1280.0 / v["size"][0]
            items.append(("random", name, pose_fn(p) if pose_fn else p, "train"))            # pose_fn: re-express under a new base
        if n_near: log(f"  {n_near} YES frames skipped: within {keep_away_s:.0f} s of a held-back moment")
    zips = {s: zipfile.ZipFile(p) for s, p in frame_zips.items()}; poses = {}; count = {"train": 0, "val": 0}; missing = 0
    for sub in ("images/train", "images/val", "masks/train", "masks/val", "images_full/val"): os.makedirs(f"{out_dir}/{sub}", exist_ok=True)
    for s, name, pose, split in items:
        try: img = cv2.imdecode(np.frombuffer(zips[s].read(name), np.uint8), cv2.IMREAD_COLOR)
        except KeyError: missing += 1; continue
        if full_val and split == "val": cv2.imwrite(f"{out_dir}/images_full/val/{s}_{name[:-4]}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA); m = render_mask(cam, pose, w, h, near_m=near_m)
        if ((m > 0) & (m != IGNORE)).sum() < 50: continue                          # no pitch lines in view: nothing to learn
        key = f"{s}_{name[:-4]}"
        cv2.imwrite(f"{out_dir}/images/{split}/{key}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92]); cv2.imwrite(f"{out_dir}/masks/{split}/{key}.png", m)
        poses[key] = {"pose": [float(v) for v in pose], "split": split, "source": s, "frame": name,
                      "clicks": clicks.get(s, {}).get(name, {}).get("pairs")}; count[split] += 1
    json.dump({"camera": cam, "size": list(size), "classes": CLASSES, "poses": poses}, open(f"{out_dir}/poses.json", "w"))
    log(f"line dataset: {count['train']} training frames, {count['val']} held-back clicked frames ({missing} missing from zips) -> {out_dir}")
    return count

def preview(ds_dir, n=6, split="train", seed=0):
    """a few label overlays stacked into one small JPEG (bytes), to check labels sit on the painted lines"""
    import os
    names = sorted(os.listdir(f"{ds_dir}/images/{split}")); rng = np.random.RandomState(seed)
    pick = [names[i] for i in sorted(rng.choice(len(names), min(n, len(names)), replace=False))]; rows = []
    for f in pick:
        img = cv2.imread(f"{ds_dir}/images/{split}/{f}"); m = cv2.imread(f"{ds_dir}/masks/{split}/{f[:-4]}.png", cv2.IMREAD_GRAYSCALE)
        o = overlay(img, m, 0.55); cv2.putText(o, f[:-4], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2); rows.append(np.hstack([img, o]))
    return cv2.imencode(".jpg", cv2.resize(np.vstack(rows), None, fx=0.6, fy=0.6), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

def draw_pose(img, camera, pose, colour=None, thick=2):
    """the model's lines through a pose, coloured by class (or one colour), on a copy of img"""
    out = img.copy(); h, w = img.shape[:2]
    for k, S in projected_segments(camera, pose, w, h).items():
        for x0, y0, x1, y1 in S:
            ok, p1, p2 = cv2.clipLine((0, 0, w, h), (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))))
            if ok: cv2.line(out, p1, p2, colour or COLOURS[k], thick, cv2.LINE_AA)
    return out

def snap(img, camera, pose, rounds=3, L=L_DEF, W=W_DEF):
    """final polish on the actual painted white pixels at full size (the old line refine, which is precise once it starts
    from the right place, and limited to small moves so it can't jump). pose f is for 1280 wide; returns the same."""
    from . import fccam as FC
    FC.set_base_tilt(*camera["base_tilt"]); w = img.shape[1]; k = w / 1280.0
    p = np.array([pose[0], pose[1], pose[2], pose[3] * k], float)
    for _ in range(rounds): p = FC.refine(img, camera["C"], L, W, p)
    return [float(p[0]), float(p[1]), float(p[2]), float(p[3] / k)]
