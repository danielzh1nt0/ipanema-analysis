"""B4 (1 Oct): grow the ball answer key without clicking.
A frame becomes a candidate key moment when the picker's ball, RF-DETR's top guess and WASB's top guess all sit on the
same spot with good scores. Claude then checks every candidate by eye on picture sheets (tools/b4_sheet.py)."""
import numpy as np


def _top(cands, i):
    v = cands.get(i)
    return max(v, key=lambda z: z[2]) if v else None


def agreed(i, pick, rf, wasb, rf_min=0.4, wasb_min=0.5, r_px=10.0):
    """True when the picker, RF-DETR's top guess and WASB's top guess agree (within r_px) at frame i"""
    p, a, b = pick.get(i), _top(rf, i), _top(wasb, i)
    if p is None or a is None or b is None or a[2] < rf_min or b[2] < wasb_min:
        return False
    return np.hypot(a[0] - p[0], a[1] - p[1]) <= r_px and np.hypot(b[0] - p[0], b[1] - p[1]) <= r_px


def select(pick, rf, wasb, n, gap=15, steady=3, avoid=(), avoid_px=5, **kw):
    """frames for the auto key: agreed at the frame and on `steady` frames each side (so it's not a one-frame fluke),
    at least `gap` frames apart, not within avoid_px frames of an existing key frame. Returns [(frame, x, y, rf, wasb)]."""
    ok = np.array([agreed(i, pick, rf, wasb, **kw) for i in range(n)], bool)
    avoid = sorted(avoid); out = []; last = -10 ** 9
    for i in range(n):
        if not ok[i] or i - last < gap: continue
        if not ok[max(0, i - steady):i + steady + 1].all(): continue
        if any(abs(i - a) <= avoid_px for a in avoid): continue
        out.append((i, float(pick[i][0]), float(pick[i][1]), float(_top(rf, i)[2]), float(_top(wasb, i)[2])))
        last = i
    return out


def tile(frame, x, y, half=48, scale=2, label=""):
    """crop around (x, y), enlarged; four short ticks mark the guess without covering the ball"""
    import cv2
    h, w = frame.shape[:2]; x, y = int(round(x)), int(round(y))
    pad = cv2.copyMakeBorder(frame, half, half, half, half, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    c = pad[y:y + 2 * half, x:x + 2 * half]
    c = cv2.resize(c, (2 * half * scale, 2 * half * scale), interpolation=cv2.INTER_CUBIC)
    m = half * scale
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(c, (m + dx * 16, m + dy * 16), (m + dx * 28, m + dy * 28), (0, 255, 255), 2)
    if label:
        cv2.putText(c, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(c, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return c


def sheet(tiles, cols=6):
    """grid of equal-size tiles (blank-filled)"""
    th, tw = tiles[0].shape[:2]; rows = -(-len(tiles) // cols)
    out = np.full((rows * th, cols * tw, 3), 255, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols); out[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    return out
