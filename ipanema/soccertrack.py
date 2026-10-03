"""B3 (3 Oct): SoccerTrack v2 (HF atomscott/soccertrack-v2, data CC BY 4.0, code MIT) ball labels in PIXELS.

The ball in SoccerTrack v2 is only given as pitch coordinates (BePro tracking XML, loc in the unit square). The chain
to the 4K fisheye panorama follows the dataset's own scripts/calibration/project_tracking_to_image.py:

    loc -> metres (x*105, y*68) -> cv2.fisheye.projectPoints(rvec, tvec, K, D) -> pixels

with K, D, rvec, tvec from cv2.fisheye.calibrate on the match's pitch keypoints (one fixed camera per match).
Known data problems handled as in that script (do not "fix" them):
  * 132831 ships two swapped keypoints -> corrected copy in reference/soccertrack/132831_keypoints.json
  * y is inverted for 132831 and 132877 (players), and in 132877 the BALL is not inverted (disagrees with its players)
  * 132831's ball sits at exactly (0.5, 0.5) in places -> placeholder, dropped
  * video frame 0 = the first XML frameNumber of that period (offset differs per match)
The tracked ball has no height, so a ball in the air projects onto the grass below it. We keep GROUND balls only:
at a player's feet (<= FEET_M from the nearest player) or rolling (slow, steady, no gap around the frame).
Then a small snap looks for the ball blob near the projection (projection error is ~9-30 px RMS in 4K).
"""
from __future__ import annotations
import json, os, re
import numpy as np

PITCH_W, PITCH_H = 105.0, 68.0
TRACKING_Y_FLIP = {"132831": True, "132877": True}
BALL_Y_FLIP_DISAGREES = {"132877"}
KEYPOINT_SWAPS = {"132831": [("(88.5,13.84)", "(105,54.16)")]}
MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578", "128057", "128058", "132831", "132877"]
FEET_M = 2.0           # ball within 2 m of a player = at feet (ground)
ROLL_MAX_MS = 12.0     # rolling ball: speed below this over the whole +-ROLL_WIN window ...
ROLL_STEADY = 4.0      # ... and speed never jumps by more than this between frames (a kick/bounce/flight)
ROLL_WIN = 5           # frames each side that must all be tracked
BALL_D_M = 0.22

_FRAME_RE = re.compile(r'<frame[^>]*\bframeNumber="(\d+)"[^>]*\beventPeriod="(\w+)"')
_FRAME_RE2 = re.compile(r'<frame[^>]*\beventPeriod="(\w+)"[^>]*\bframeNumber="(\d+)"')
_ENT_RE = re.compile(r'<(player|ball)\s+playerId="([^"]+)"\s+loc="\[([-\d.eE]+),\s*([-\d.eE]+)\]"')


def parse_xml(path, periods=("FIRST_HALF", "SECOND_HALF")):
    """stream the tracker XML -> {period: {"frames": int array, "ball": (n,2) loc or nan, "players": list of (k,2)}}"""
    out = {p: {"frames": [], "ball": [], "players": []} for p in periods}
    cur = None; per = None; ball = None; pl = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            if "<frame" in line:
                m = _FRAME_RE.search(line)
                if m: cur, per = int(m.group(1)), m.group(2)
                else:
                    m = _FRAME_RE2.search(line)
                    if m: cur, per = int(m.group(2)), m.group(1)
                ball = None; pl = []
            if cur is not None:
                for e in _ENT_RE.finditer(line):
                    xy = (float(e.group(3)), float(e.group(4)))
                    if e.group(1) == "ball": ball = xy
                    else: pl.append(xy)
            if "</frame>" in line and cur is not None:
                if per in out:
                    o = out[per]; o["frames"].append(cur); o["ball"].append(ball if ball else (np.nan, np.nan)); o["players"].append(np.array(pl, float).reshape(-1, 2))
                cur = None
    for p in periods:
        o = out[p]; o["frames"] = np.array(o["frames"], int); o["ball"] = np.array(o["ball"], float).reshape(-1, 2)
    return out


def to_metres(loc, match, kind):
    """unit-square loc -> pitch metres with this match's y convention"""
    loc = np.asarray(loc, float).reshape(-1, 2)
    flip = TRACKING_Y_FLIP.get(str(match), False)
    if kind == "ball" and str(match) in BALL_Y_FLIP_DISAGREES: flip = not flip
    y = (1.0 - loc[:, 1]) if flip else loc[:, 1]
    return np.stack([loc[:, 0] * PITCH_W, y * PITCH_H], 1)


def load_keypoints(path, match, corrected=None):
    """pitch metres (n,2), image pixels (n,2), source; a corrected copy wins over the shipped file"""
    src = "dataset"
    if corrected and os.path.exists(corrected): path = corrected; src = "corrected copy"
    d = json.load(open(path)); keys = list(d)
    pitch = np.array([[*map(float, k.strip("()").split(","))] for k in keys], float)
    image = np.array(list(d.values()), float)
    if src == "dataset":
        for a, b in KEYPOINT_SWAPS.get(str(match), []):
            if a in d and b in d:
                i, j = keys.index(a), keys.index(b); image[[i, j]] = image[[j, i]]; src = "dataset + swap"
    return pitch, image, src


def _reproj(cal, pitch, image):
    return np.sqrt(np.mean(np.sum((project(cal, pitch) - image) ** 2, 1)))


def calibrate(pitch, image, w, h):
    """fisheye camera from one view of the pitch keypoints, the dataset's model (k1, k2 free; k3, k4, skew fixed).
    cv2.fisheye.calibrate as in the dataset's script, then a least-squares polish of the same reprojection error
    (OpenCV 5's fisheye.calibrate alone converges badly on one planar view; OpenCV 4.10, the dataset's, does not).
    Raises cv2.error when no start converges (like CALIB_CHECK_COND refusing 132831's shipped keypoints)."""
    import cv2
    from scipy.optimize import least_squares
    fe = cv2.fisheye
    def F(name, val): return getattr(fe, name, val)                     # OpenCV 5 dropped the named fisheye flags
    base = F("CALIB_RECOMPUTE_EXTRINSIC", 2) + F("CALIB_FIX_SKEW", 8) + F("CALIB_FIX_K3", 64) + F("CALIB_FIX_K4", 128)
    crit = (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS, 100, 1e-6)
    objp = np.concatenate([pitch, np.zeros((len(pitch), 1))], 1).astype(np.float64).reshape(1, -1, 3)
    imgp = image.astype(np.float64).reshape(1, -1, 2)
    starts = []
    for f0 in (None, w / np.pi, w / 4, w / 2.5, w / 2, w / 1.5, w):
        K0 = np.zeros((3, 3)) if f0 is None else np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1.0]])
        fl = base + F("CALIB_CHECK_COND", 4) if f0 is None else base + F("CALIB_USE_INTRINSIC_GUESS", 1)
        try:
            _, K, D, rv, tv = cv2.fisheye.calibrate([objp], [imgp], (w, h), K0, np.zeros((4, 1)), None, None, flags=fl, criteria=crit)
            starts.append(dict(K=K, D=D, rvec=np.asarray(rv[0], float).reshape(3, 1), tvec=np.asarray(tv[0], float).reshape(3, 1)))
        except cv2.error:
            continue
    if not starts: raise cv2.error("fisheye calibration refused for every start")

    def unpack(v):
        K = np.array([[v[0], 0, v[1]], [0, v[0] * v[2], v[3]], [0, 0, 1.0]])
        return dict(K=K, D=np.array([[v[4]], [v[5]], [0.0], [0.0]]), rvec=v[6:9].reshape(3, 1), tvec=v[9:12].reshape(3, 1))
    def res(v): return (project(unpack(v), pitch) - image).ravel()
    best = None
    for c in starts:
        K = c["K"]; v0 = np.r_[K[0, 0], K[0, 2], K[1, 1] / K[0, 0], K[1, 2], c["D"].ravel()[:2], c["rvec"].ravel(), c["tvec"].ravel()]
        if not np.isfinite(v0).all(): continue
        try: r = least_squares(res, v0, method="lm", max_nfev=4000)
        except Exception: continue
        e = float(np.sqrt(np.mean(r.fun.reshape(-1, 2) ** 2 * 2)))
        if best is None or e < best[0]: best = (e, r.x)
    if best is None: raise cv2.error("fisheye calibration did not converge")
    cal = unpack(best[1]); cal["rms"] = best[0]
    return cal


def project(cal, metres):
    import cv2
    m = np.asarray(metres, float).reshape(-1, 2)
    obj = np.concatenate([m, np.zeros((len(m), 1))], 1).astype(np.float64).reshape(1, -1, 3)
    pts, _ = cv2.fisheye.projectPoints(obj, cal["rvec"], cal["tvec"], cal["K"], cal["D"])
    return pts.reshape(-1, 2)


def ball_px_size(cal, metres):
    """apparent ball diameter in pixels at a pitch point (projected 0.22 m along x and y, the larger)"""
    m = np.asarray(metres, float).reshape(-1, 2); r = BALL_D_M / 2
    a = project(cal, np.concatenate([m - [r, 0], m + [r, 0], m - [0, r], m + [0, r]]))
    n = len(m); dx = np.linalg.norm(a[:n] - a[n:2 * n], axis=1); dy = np.linalg.norm(a[2 * n:3 * n] - a[3 * n:], axis=1)
    return np.maximum(dx, dy)


def ground_candidates(per, match, fps=25.0):
    """per-frame features + ground flag for one period. Returns dict of arrays aligned with per['frames']."""
    loc = per["ball"]; n = len(loc)
    ok = np.isfinite(loc).all(1)
    ok &= ~((np.abs(loc[:, 0] - 0.5) < 1e-6) & (np.abs(loc[:, 1] - 0.5) < 1e-6))     # 132831 placeholder
    bm = np.full((n, 2), np.nan); bm[ok] = to_metres(loc[ok], match, "ball")
    inside = ok & (bm[:, 0] >= -0.5) & (bm[:, 0] <= PITCH_W + 0.5) & (bm[:, 1] >= -0.5) & (bm[:, 1] <= PITCH_H + 0.5)
    dpl = np.full(n, np.inf)
    for i in np.where(ok)[0]:
        p = per["players"][i]
        if len(p): dpl[i] = np.linalg.norm(to_metres(p, match, "player") - bm[i], axis=1).min()
    sp = np.full(n, np.nan)
    fr = per["frames"]; consec = np.r_[False, np.diff(fr) == 1]
    v = np.linalg.norm(np.diff(bm, axis=0), axis=1) * fps; sp[1:] = np.where(consec[1:], v, np.nan)
    feet = inside & (dpl <= FEET_M)
    rolling = np.zeros(n, bool)
    for i in np.where(inside & ~feet)[0]:
        a, b = i - ROLL_WIN + 1, i + ROLL_WIN + 1
        if a < 1 or b > n: continue
        w = sp[a:b]
        if np.isfinite(w).all() and w.max() <= ROLL_MAX_MS and np.abs(np.diff(w)).max() <= ROLL_STEADY: rolling[i] = True
    return dict(metres=bm, d_player=dpl, speed=sp, feet=feet, rolling=rolling, ground=feet | rolling, inside=inside)


def pick_frames(frames_idx, ok_mask, kind, n_want, min_gap, seed=3):
    """spread picks over the half: n_want frames from ok_mask, at least min_gap apart, about half feet / half rolling"""
    rng = np.random.default_rng(seed); cand = np.where(ok_mask)[0]
    if not len(cand): return []
    rng.shuffle(cand)
    # interleave kinds so both are represented
    a = [i for i in cand if kind[i] == "rolling"]; b = [i for i in cand if kind[i] != "rolling"]
    order = [x for pair in zip(a, b) for x in pair] + a[len(b):] + b[len(a):]
    chosen = []
    for i in order:
        if all(abs(frames_idx[i] - frames_idx[j]) >= min_gap for j in chosen): chosen.append(i)
        if len(chosen) >= n_want: break
    return sorted(chosen)


def snap(img, xy, radius, ball_px):
    """look for a ball-like blob (bright/dark small spot vs its ring) within `radius` px of xy.
    Returns (x, y, contrast, found)."""
    import cv2
    h, w = img.shape[:2]; x0, y0 = int(round(xy[0])), int(round(xy[1])); R = int(radius) + 8
    xa, xb, ya, yb = max(0, x0 - R), min(w, x0 + R + 1), max(0, y0 - R), min(h, y0 + R + 1)
    if xb - xa < 9 or yb - ya < 9: return float(xy[0]), float(xy[1]), 0.0, False
    g = cv2.cvtColor(img[ya:yb, xa:xb], cv2.COLOR_BGR2GRAY).astype(np.float32)
    s = max(1.0, ball_px / 2.0)
    inner = cv2.GaussianBlur(g, (0, 0), s * 0.6); outer = cv2.GaussianBlur(g, (0, 0), s * 2.2)
    resp = inner - outer                                                       # white ball on grass = positive
    yy, xx = np.mgrid[ya:yb, xa:xb]; resp[(xx - xy[0]) ** 2 + (yy - xy[1]) ** 2 > radius ** 2] = -1e9
    k = int(np.argmax(resp)); py, px = divmod(k, resp.shape[1]); c = float(resp[py, px])
    noise = float(np.std(resp[resp > -1e8])) + 1e-6
    found = c >= 12.0 and c / noise >= 3.0
    r = max(2, int(round(s * 1.5))); y1, y2, x1, x2 = max(0, py - r), py + r + 1, max(0, px - r), px + r + 1
    wgt = np.clip(resp[y1:y2, x1:x2] - 0.3 * c, 0, None)                        # sub-pixel: centroid of the peak's top
    if wgt.sum() > 0:
        gy, gx = np.mgrid[y1:y1 + wgt.shape[0], x1:x1 + wgt.shape[1]]; px, py = (gx * wgt).sum() / wgt.sum(), (gy * wgt).sum() / wgt.sum()
    return float(xa + px), float(ya + py), c, bool(found)


def crop(img, xy, size):
    """size x size crop centred on xy, zero padded at the borders"""
    h, w = img.shape[:2]; r = size // 2; x, y = int(round(xy[0])), int(round(xy[1]))
    out = np.zeros((size, size, 3), np.uint8)
    xa, xb, ya, yb = x - r, x - r + size, y - r, y - r + size
    sa, sb, ta, tb = max(0, xa), min(w, xb), max(0, ya), min(h, yb)
    if sb > sa and tb > ta: out[ta - ya:tb - ya, sa - xa:sb - xa] = img[ta:tb, sa:sb]
    return out
