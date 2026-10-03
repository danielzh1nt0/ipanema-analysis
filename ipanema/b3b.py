"""B3b (3 Oct): check the SoccerTrack v2 ball labels (B3, results/free/b3/labels.json) with our own ball finder.

The B3 projection of the dataset's tracked ball lands 0.5-2 m (10-40 px) from the real ball. Here the RF-DETR finder
(B7, the app's finder) looks in a window around each projection; a label is kept only when the finder is confident
inside RADIUS px of the projection and no second guess there is nearly as strong (two balls / ball vs shoe = unsure).

Pure helpers (no model, no video) so they can be tested: window cutting with the way back to frame pixels, the best
guess near the projection, the keep rule.
"""
import numpy as np, cv2

SIZE = 640          # RF-DETR tile, as in ipanema/ballrf.py
RADIUS = 60.0       # px in the video frame around the projection (B3: ball typically 10-40 px off)
CONF = 0.5          # keep only confident hits
RIVAL = 0.6         # a second guess >= RIVAL * best, more than AGREE px away, inside RADIUS -> unsure
AGREE = 6.0         # px: two guesses (or two scales) this close are the same ball


def window(img, cx, cy, size=SIZE, scale=1.0):
    """cut a (size/scale)-px square around (cx, cy), pushed inside the image, resized to size x size.
    Returns (crop, (x0, y0, scale)); frame = x0 + crop_x / scale."""
    h, w = img.shape[:2]; side = int(round(size / scale))
    sw, sh = min(side, w), min(side, h)
    x0 = int(np.clip(round(cx - sw / 2), 0, w - sw)); y0 = int(np.clip(round(cy - sh / 2), 0, h - sh))
    c = img[y0:y0 + sh, x0:x0 + sw]
    if (sw, sh) != (side, side):                                     # image smaller than the window: pad with grey
        pad = np.full((side, side, 3), 114, np.uint8); pad[:sh, :sw] = c; c = pad
    if scale != 1.0: c = cv2.resize(c, (size, size), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(c), (x0, y0, float(scale))


def to_frame(dets, t):
    """[(x, y, conf)] in crop pixels -> frame pixels"""
    x0, y0, s = t
    return [(x0 + x / s, y0 + y / s, float(c)) for x, y, c in dets]


def near(dets, xy, radius=RADIUS):
    """guesses within radius of xy, best first"""
    out = [d for d in dets if np.hypot(d[0] - xy[0], d[1] - xy[1]) <= radius]
    return sorted(out, key=lambda d: -d[2])


def merge_scales(per_scale, agree=AGREE):
    """{scale: [(x, y, conf)] near the projection} -> one list: guesses seen at several scales are joined (max conf,
    position of the best), with 'n' = how many scales saw it"""
    allg = sorted(((x, y, c, s) for s, g in per_scale.items() for x, y, c in g), key=lambda d: -d[2]); out = []
    for x, y, c, s in allg:
        for o in out:
            if np.hypot(o["x"] - x, o["y"] - y) <= agree:
                o["scales"].add(s); break
        else: out.append(dict(x=float(x), y=float(y), conf=float(c), scales={s}))
    for o in out: o["n"] = len(o["scales"]); o["scales"] = sorted(o["scales"])
    return out


def decide(guesses, conf=CONF, rival=RIVAL, agree=AGREE):
    """guesses (merge_scales output, best first) -> (keep, reason)"""
    if not guesses: return False, "no guess"
    b = guesses[0]
    if b["conf"] < conf: return False, "weak"
    for g in guesses[1:]:
        if g["conf"] >= rival * b["conf"] and np.hypot(g["x"] - b["x"], g["y"] - b["y"]) > agree: return False, "two guesses"
    return True, "kept"
