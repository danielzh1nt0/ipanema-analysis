"""S3 (3 Oct): pasted balls - real ball patches cut from checked ball crops (Daniel's SFK-BP clicks + the K1 balls of
4 more grounds), shrunk to far-side sizes (4-8 px), matched to the local light, blurred and pasted onto our own grass
frames, as extra training pictures for the ball finder. Only pictures + labels here; no training (the sheets are checked
by eye first). Free, CPU.

Steps: cut_patch (ball + soft mask from a 32x32 crop, only clean isolated balls), size_at (ball size from the players'
box heights in that frame: a ball is ~0.12 of a player's height), spots (on the pitch's grass, away from people and the
frame edge, biased to the far side), paste (light match, blur, optional motion smear, a little noise)."""
import numpy as np, cv2

BALL_PER_PLAYER = 0.12      # 22 cm ball / ~1.8 m player box (boxes run a bit taller than the player: keeps it small)


def cut_patch(crop, diff_thr=22.0, min_d=4.0, max_d=16.0, ring_dirty=0.25, centre_r=6):
    """crop: (S, S, 3) uint8 BGR around a checked ball. Returns dict(rgb (h,w,3) float, alpha (h,w) 0..1, d diameter px,
    grass_L lightness of the grass around it) or None when the ball is not clean (legs, lines, crowd, not round)."""
    S = crop.shape[0]; lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(float)
    yy, xx = np.mgrid[:S, :S]; r = np.hypot(yy - (S - 1) / 2, xx - (S - 1) / 2)
    ring = r >= S / 2 - 3
    g = np.median(lab[ring], axis=0)
    diff = np.linalg.norm(lab - g, axis=2)
    if (diff[ring] > diff_thr).mean() > ring_dirty: return None            # lines, legs or crowd near the ball
    m = (diff > diff_thr).astype(np.uint8)
    n, lb, st, cen = cv2.connectedComponentsWithStats(m)
    cand = [i for i in range(1, n) if r[lb == i].min() <= centre_r]
    if not cand: return None
    i = max(cand, key=lambda k: st[k, cv2.CC_STAT_AREA])
    x, y, w, h, area = st[i]
    d = float(np.sqrt(4 * area / np.pi))
    if not (min_d <= d <= max_d) or max(w, h) > 1.6 * min(w, h) + 1: return None   # not ball-sized / not round
    if np.percentile(lab[lb == i][:, 0], 90) < g[0] + 25: return None       # no bright panel: a shadow or shoe, not the ball
    if area < 0.45 * w * h: return None
    if x == 0 or y == 0 or x + w >= S or y + h >= S: return None
    comp = (lb == i)
    soft = np.clip((diff - 0.5 * diff_thr) / diff_thr, 0, 1) * cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8))
    p = 1
    x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(S, x + w + p), min(S, y + h + p)
    gb = np.median(crop[ring].reshape(-1, 3).astype(float), axis=0)
    return dict(rgb=crop[y0:y1, x0:x1].astype(float), alpha=soft[y0:y1, x0:x1].astype(float), d=d, grass_L=float(g[0]),
                grass_bgr=gb)


def size_at(boxes, y, frame_h):
    """expected ball diameter (px) at image row y, from the people in this frame: box height grows ~linearly with row.
    Fallback (too few people): 4 px at the top of the frame to 10 px at the bottom."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    if len(b) >= 4:
        feet, hgt = b[:, 3], b[:, 3] - b[:, 1]
        A = np.vstack([feet, np.ones_like(feet)]).T
        k, c = np.linalg.lstsq(A, hgt, rcond=None)[0]
        if k > 0: return float(max(2.0, BALL_PER_PLAYER * (k * y + c)))
    return float(4 + 6 * y / frame_h)


def spots(frame, boxes, n, rng, d_lo=4.0, d_hi=8.0, far_bias=2.0, edge=40, gap=1.0, grass_d=16.0, tries=4000,
          mix=(0.5, 0.25, 0.25)):
    """n spots (x, y, d, kind) at rows where the expected ball size is d_lo..d_hi px (the far half), drawn with more
    weight further away (far_bias). kind (shares in mix): 'open' = plain grass, away from people (one box width) and
    lines; 'line' = on the pitch with a line in the window allowed; 'feet' = beside a player's feet, not covering him
    (the finder's weak spot: balls at feet). The blind check showed pasted balls on empty grass give themselves away."""
    from .kits import pitch_top, grass_lab
    H, W = frame.shape[:2]; top = pitch_top(frame); g = grass_lab(frame)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(float)
    grass = (np.hypot(lab[..., 1] - g[1], lab[..., 2] - g[2]) <= grass_d)
    b = np.asarray(boxes, float).reshape(-1, 4); out = []
    bw = b[:, 2] - b[:, 0] if len(b) else np.zeros(0)
    for _ in range(tries):
        if len(out) >= n: break
        kind = ["open", "line", "feet"][int(rng.choice(3, p=np.asarray(mix) / np.sum(mix)))]
        if kind == "feet":
            if not len(b): continue
            j = int(rng.integers(len(b))); x1, y1, x2, y2 = b[j]
            d = size_at(b, y2, H)
            side = 1 if rng.random() < 0.5 else -1
            x = int(round((x2 if side > 0 else x1) + side * (d / 2 + 1 + rng.uniform(0, 0.6) * bw[j])))
            y = int(round(y2 - rng.uniform(0, 0.6) * d))
            if not (edge <= x < W - edge and edge <= y < H - edge) or y < top[x] + 10: continue
        else:
            x = int(rng.integers(edge, W - edge)); t = int(top[x]) + 10
            if t >= H - edge: continue
            u = rng.random() ** far_bias; y = int(t + u * (H - edge - t))      # u small = near the far touchline
            d = size_at(b, y, H)
        if not (d_lo <= d <= d_hi): continue
        if len(b):                                                            # never on top of a person
            r = d / 2 + 1
            if np.any((x + r > b[:, 0]) & (x - r < b[:, 2]) & (y + r > b[:, 1]) & (y - r < b[:, 3])): continue
            if kind != "feet" and np.any((x > b[:, 0] - gap * bw) & (x < b[:, 2] + gap * bw) &
                                         (y > b[:, 1] - 10) & (y < b[:, 3] + 15)): continue
        w = grass[y - 8:y + 9, x - 8:x + 9]
        if kind == "open" and w.mean() < 0.97: continue                       # lines, shadows of people, off-pitch
        if kind != "open" and (w.mean() < 0.5 or grass[y - 2:y + 3, x - 2:x + 3].mean() < 0.3): continue
        if kind == "line" and w.mean() > 0.95: continue                       # no line there: not a line spot
        if any(abs(x - o[0]) < 40 and abs(y - o[1]) < 40 for o in out): continue
        out.append((x, y, float(d), kind))
    return out


def _motion_kernel(L, ang):
    k = np.zeros((L * 2 + 1, L * 2 + 1), np.float32); c = L
    dx, dy = np.cos(ang), np.sin(ang)
    for t in np.linspace(-L / 2, L / 2, 4 * L + 1):
        k[int(round(c + t * dy)), int(round(c + t * dx))] = 1
    return k / k.sum()


def paste(frame, patch, x, y, d, rng, blur=(0.2, 0.5), smear=(2, 4), smear_p=0.3, sharpen=0.0, noise=2.0, light=True,
          shadow_p=0.5, codec_q=(60, 85)):
    """paste one patch with diameter d (px) centred at (x, y) into frame (BGR uint8, modified copy returned) and the
    label box [x1, y1, x2, y2]. light: scale the patch so its grass-to-ball contrast matches this spot's grass
    (night and shade pitches are darker than the sunny source crops). Graded blind (results/ball/s3/README.md): heavy
    blur/smear made pasted balls washed out and easy to spot, so blur is light, a smear only on 30% (a moving ball), and
    then crisp 'sticker' balls were easy to spot too: now a soft ground shadow, the codec's half-resolution colour and a
    JPEG round trip of the small window around the ball."""
    s = d / patch["d"]; rgb, a = patch["rgb"], patch["alpha"]
    if rng.random() < 0.5: rgb, a = rgb[:, ::-1].copy(), a[:, ::-1].copy()   # mirrored only: light comes from above
    h, w = max(2, int(round(rgb.shape[0] * s))), max(2, int(round(rgb.shape[1] * s)))
    rgb = cv2.resize(rgb.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    a = cv2.resize(a.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    out = frame.copy(); H, W = frame.shape[:2]
    P = int(max(h, w)) + 8; x0, y0 = int(x) - P, int(y) - P
    if x0 < 0 or y0 < 0 or x0 + 2 * P + 1 > W or y0 + 2 * P + 1 > H: return out, None
    win = out[y0:y0 + 2 * P + 1, x0:x0 + 2 * P + 1].astype(np.float32)
    if light:                                     # the light's colour and strength at this spot (floodlights are yellow)
        tg = np.median(win.reshape(-1, 3), axis=0); sg0 = np.asarray(patch.get("grass_bgr", tg), float)
        rgb = rgb * np.clip(np.sqrt((tg + 10) / (sg0 + 10)), 0.7, 1.4)
    layer = np.zeros_like(win); al = np.zeros(win.shape[:2], np.float32)
    py, px = P - h // 2, P - w // 2
    layer[py:py + h, px:px + w] = rgb; al[py:py + h, px:px + w] = a
    L = int(rng.integers(smear[0], smear[1] + 1)) if rng.random() < smear_p else 0
    if L >= 2:                                                                # a moving ball smears along its path
        k = _motion_kernel(L, float(rng.uniform(0, np.pi)))
        layer = cv2.filter2D(layer * al[..., None], -1, k); al = cv2.filter2D(al, -1, k)
        layer = layer / np.maximum(al[..., None], 1e-4)
    sg = float(rng.uniform(*blur))
    if sg > 0.05:
        layer = cv2.GaussianBlur(layer * al[..., None], (0, 0), sg); al = cv2.GaussianBlur(al, (0, 0), sg)
        layer = layer / np.maximum(al[..., None], 1e-4)
    al = np.clip(al, 0, 1)[..., None]
    if rng.random() < shadow_p:                                               # a ground ball's soft shadow just below it
        sh = np.zeros(win.shape[:2], np.float32)
        cv2.ellipse(sh, (P, int(P + 0.35 * d)), (max(1, int(0.55 * d)), max(1, int(0.22 * d))), 0, 0, 360, 1.0, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), max(0.6, 0.25 * d)) * float(rng.uniform(0.15, 0.3))
        win = win * (1 - sh[..., None])
    comp = win * (1 - al) + layer * al
    if sharpen > 0:                                                           # Veo-like sharpening, only around the ball
        m = cv2.dilate((al[..., 0] > 0.05).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)[..., None]
        comp = comp + sharpen * m * (comp - cv2.GaussianBlur(comp, (0, 0), 1.0))
    if noise > 0: comp = comp + rng.normal(0, noise, comp.shape) * al
    if codec_q:                                   # the video codec: colour at half resolution (4:2:0) + compression
        ycc = cv2.cvtColor(np.clip(comp, 0, 255).astype(np.uint8), cv2.COLOR_BGR2YCrCb)
        hh, ww = ycc.shape[:2]
        for c in (1, 2): ycc[..., c] = cv2.resize(cv2.resize(ycc[..., c], (ww // 2, hh // 2), interpolation=cv2.INTER_AREA), (ww, hh), interpolation=cv2.INTER_LINEAR)
        q = int(rng.integers(codec_q[0], codec_q[1] + 1))
        comp = cv2.imdecode(cv2.imencode(".jpg", cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1).astype(np.float32)
    out[y0:y0 + 2 * P + 1, x0:x0 + 2 * P + 1] = np.clip(comp, 0, 255).astype(np.uint8)
    r = d / 2 + L / 2
    return out, [float(x - r), float(y - r), float(x + r), float(y + r)]


# ---------------- S3b (3 Oct): pasting inside the training crops of the ball finder (kaggle/ballfinder_paste.py)
def _crop_t(name):
    """SFK-BP crop file name 't01234.5.jpg' -> 1234.5 s (None when not a time name)"""
    try: return float(str(name)[1:-4])
    except ValueError: return None


def load_patches(repo=".", skip_t=None):
    """clean ball patches from the checked ball crops in the repo (as tools/s3_paste.py): SFK-BP TRAIN clicks (skip_t(t)
    True = left out, e.g. the test clip's window and +-1 s around exam frames) + the K1 checked balls of 4 more grounds."""
    out = []
    d = np.load(f"{repo}/results/volume/cache/ballcrops/SFKBP1109_crops.npz", allow_pickle=True); X, M = d["X"], d["meta"]
    for i in range(len(M)):
        if M[i][6] == 1 and M[i][1] == "train":
            t = _crop_t(M[i][0])
            if skip_t is not None and (t is None or skip_t(t)): continue
            p = cut_patch(X[i][1])
            if p: p.update(sid=f"sfk:{i}"); out.append(p)
    k = np.load(f"{repo}/results/free/k1b/k1_crops.npz", allow_pickle=True); KX, KM = k["X"], k["meta"]
    for i in range(len(KM)):
        if KM[i][6] == 1 and int(KM[i][8]) == 0:
            p = cut_patch(KX[i][1])
            if p: p.update(sid=f"k1:{i}"); out.append(p)
    return out


def far_spots(frame, boxes, rng, n=12, d_lo=4.0, d_hi=8.0):
    """candidate far spots of one frame (d_lo..d_hi px at 1080 rows, scaled with the frame height)"""
    s = frame.shape[0] / 1080.0
    return spots(frame, boxes, n, rng, d_lo=d_lo * s, d_hi=d_hi * s)


def in_window(sp, real_xy, win, margin=24, min_gap=40):
    """the spots inside window win = (x0, y0, x1, y1) (margin px from its edge) and min_gap px from the real ball"""
    x0, y0, x1, y1 = win
    return [p for p in sp if x0 + margin <= p[0] < x1 - margin and y0 + margin <= p[1] < y1 - margin
            and (real_xy is None or np.hypot(p[0] - real_xy[0], p[1] - real_xy[1]) >= min_gap)]


def paste_in_window(frame, boxes, real_xy, win, patches, rng, d_lo=4.0, d_hi=8.0, margin=24, min_gap=40, n_try=8, cands=None):
    """paste ONE far ball (d_lo..d_hi px at 1080 rows, scaled with the frame height) inside the window win = (x0, y0, x1, y1)
    of frame, never on a person nor within min_gap px of the real ball real_xy (None = no real ball). Returns
    (frame copy, dict(x, y, d, kind, src)) or (frame, None) when no far spot falls inside the window."""
    sp = cands if cands is not None else far_spots(frame, boxes, rng, n_try, d_lo, d_hi)
    ok = in_window(sp, real_xy, win, margin, min_gap)
    if not ok or not patches: return frame, None
    x, y, d, kind = ok[0]; p = patches[int(rng.integers(len(patches)))]
    out, box = paste(frame, p, x, y, d, rng)
    if box is None: return frame, None
    return out, dict(x=float(x), y=float(y), d=float(d), kind=kind, src=p.get("sid"))
