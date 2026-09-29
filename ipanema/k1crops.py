"""K1 (29 Sep): ball / not-ball 3-frame crops on other matches for the second-opinion scorer (T0).
Ball positions = the click-finder check pictures Daniel answered "yes" on the 4 matches that passed his check
(results/review/<match>_clicks/items.json + answers.json; answer id = prefix + index into items). Spare balls and "no"
answers are never used. The full label sets (labels/<match>_trainset_clicks.json) live only on the Modal volume, so from
each checked ball we follow the same ball a few frames either side through WASB's peaks (small steps, stop when unsure):
~20 ball pictures per check instead of 1. Pure logic here (tested with stand-ins); the runner is tools/k1_crops.py."""
import json, os, numpy as np
from . import ballprobe as BP

GOOD = ["p15u-vs-vasalund-2026-09-20", "p09-norrviken-vs-solheim-2026-08-30", "p15u-vs-spanga-2026-09-25", "p15u-vs-djursholm-2026-09-26"]
BAD = ["p15u-vs-reymersholm-2026-09-18", "solberga-vs-p09-norrviken-2026-09-11"]      # failed Daniel's check: never used


def checked_balls(review_dir, match):
    """[{"id", "frame", "xy"}] for the check pictures Daniel answered "yes" (the match ball); spare balls / "no" left out"""
    if match in BAD: raise ValueError(f"{match} failed Daniel's check and must not be used")
    d = os.path.join(review_dir, f"{match}_clicks")
    items = json.load(open(os.path.join(d, "items.json"))); a = json.load(open(os.path.join(d, "answers.json")))
    ans = a["answers"]; skip = set(a.get("spare_ball", [])) | set(a.get("not_a_ball", []))
    idx = {int(k[-2:]): k for k in ans}
    if len(idx) != len(ans) or sorted(idx) != list(range(len(items))): raise ValueError(f"{match}: answer ids do not map onto items")
    return [{"id": idx[i], "frame": int(it["frame"]), "xy": (float(it["xy"][0]), float(it["xy"][1]))}
            for i, it in enumerate(items) if ans[idx[i]] == "yes" and idx[i] not in skip]


def follow(peaks, start_xy, window, step_px=20.0, min_score=0.1, amb_px=5.0, max_miss=1, on_ball_px=10.0):
    """peaks: {dk: [(x, y, s)]} for dk in -window..window around the checked frame (dk 0 = the check).
    Returns {dk: (x, y, s)}: the ball followed outwards from the check through WASB peaks. A step = the nearest peak with
    score >= min_score within step_px of the last position; stops on a 2nd peak about as near (unsure), or after
    max_miss frames without a peak. dk 0 is the peak on the checked spot (<= on_ball_px) if any, else the spot (score -1)."""
    near0 = [p for p in peaks.get(0, []) if np.hypot(p[0] - start_xy[0], p[1] - start_xy[1]) <= on_ball_px]
    p0 = max(near0, key=lambda p: p[2]) if near0 else (float(start_xy[0]), float(start_xy[1]), -1.0)
    out = {0: tuple(map(float, p0))}
    for sgn in (1, -1):
        last = out[0]; miss = 0
        for step in range(1, window + 1):
            dk = sgn * step
            c = sorted(((float(np.hypot(p[0] - last[0], p[1] - last[1])), p) for p in peaks.get(dk, []) if p[2] >= min_score),
                       key=lambda z: z[0])
            c = [z for z in c if z[0] <= step_px * (1 + miss)]
            if not c:
                miss += 1
                if miss > max_miss: break
                continue
            if len(c) > 1 and c[1][0] - c[0][0] < amb_px: break
            last = tuple(map(float, c[0][1])); out[dk] = last; miss = 0
    return out


def _patch(img, xy, half):
    x, y = int(round(xy[0])), int(round(xy[1])); h, w = img.shape[:2]
    if x - half < 0 or y - half < 0 or x + half >= w or y + half >= h: return None
    return img[y - half:y + half, x - half:x + half]


def blobness(img, xy, r=0, sigma=3.0):
    """bright-blob response (determinant of the Hessian of the ball-size-blurred picture, bright blobs only) at xy, or
    (best xy, response) within r px of it. A ball scores high; a pitch line scores ~0 (it only curves one way)."""
    import cv2
    x, y = int(round(xy[0])), int(round(xy[1])); m = r + int(4 * sigma) + 2; h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(w, x + m + 1), min(h, y + m + 1)
    g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.GaussianBlur(g, (0, 0), sigma)
    xx_, yy_, xy_ = (cv2.Sobel(b, cv2.CV_32F, dx, dy, ksize=3) for dx, dy in ((2, 0), (0, 2), (1, 1)))
    L = np.where(xx_ + yy_ < 0, np.sqrt(np.maximum(xx_ * yy_ - xy_ ** 2, 0)), 0) * sigma ** 2
    if r == 0: return float(L[y - y0, x - x0])
    yy, xx = np.mgrid[y0:y1, x0:x1]; L = np.where((xx - x) ** 2 + (yy - y) ** 2 <= r * r, L, -np.inf)
    j = np.unravel_index(int(np.argmax(L)), L.shape); return (float(x0 + j[1]), float(y0 + j[0])), float(L[j])


def follow_template(frames, start_xy, window, half=8, search=20, min_ncc=0.7, min_ncc_start=0.5, refine_px=10, min_blob=0.5):
    """29 Sep (the pretrained WASB misses the ball at 47 of the 65 checked spots, so following WASB peaks stops at once):
    follow the checked ball by picture matching, frame by frame. The start is moved onto the ball (the brightest ball-sized
    blob within refine_px of the check; the checks sit ~5 px above the ball's centre). Each step = the best match of the
    last ball patch within `search` px; stops when the match is weak (< min_ncc), no longer looks like the checked ball
    (< min_ncc_start), or is no longer a ball-like blob (< min_blob x the start's; e.g. stuck on a pitch line). -> {dk: (x, y)}"""
    import cv2
    start, b0 = blobness(frames[0], start_xy, refine_px) if refine_px else ((float(start_xy[0]), float(start_xy[1])), blobness(frames[0], start_xy))
    T0 = _patch(frames[0], start, half); out = {0: start}
    if T0 is None or not b0 > 0: return out
    for sgn in (1, -1):
        last, T = out[0], T0
        for step in range(1, window + 1):
            f = frames.get(sgn * step)
            if f is None: break
            x0, y0 = int(round(last[0])) - half - search, int(round(last[1])) - half - search
            if x0 < 0 or y0 < 0 or x0 + 2 * (half + search) > f.shape[1] or y0 + 2 * (half + search) > f.shape[0]: break
            r = cv2.matchTemplate(f[y0:y0 + 2 * (half + search), x0:x0 + 2 * (half + search)], T, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if not mx >= min_ncc: break
            new = (float(x0 + loc[0] + half), float(y0 + loc[1] + half)); P = _patch(f, new, half)
            if P is None or not cv2.matchTemplate(P, T0, cv2.TM_CCOEFF_NORMED)[0, 0] >= min_ncc_start: break
            if blobness(f, new) < min_blob * b0: break
            out[sgn * step] = new; last, T = new, P
    return out


def frame_labels(peaks_k, ball, n_neg, pos_px=10.0, neg_px=30.0):
    """[(x, y, s, label)] for one frame: the followed ball = 1, peaks > neg_px from it = 0 (strongest n_neg), others out"""
    negs = [(float(x), float(y), float(s), 0) for x, y, s in sorted(peaks_k, key=lambda z: -z[2])
            if np.hypot(x - ball[0], y - ball[1]) > neg_px]
    return [(ball[0], ball[1], ball[2], 1)] + negs[:n_neg]


def crops_for_check(frames, peaks, check, match, window, n_neg_check=12, n_neg_follow=6, every=2, size=32, how="wasb", snap_px=6.0):
    """frames: {dk: image} for dk in -window-1..window+1; peaks: {dk: [(x, y, s)]}. -> (X list, meta list).
    how="wasb": follow the ball through WASB peaks; "template": by picture matching (a WASB peak within snap_px is used
    instead when there is one; else the matched spot, finder score -1).
    meta = [frame id, "train", frame, x, y, finder score, label, match, dk] (same first 7 columns as the SFK-BP crops)"""
    if how == "template":
        fol = {}
        for dk, (x, y) in follow_template(frames, check["xy"], window).items():
            near = [p for p in peaks.get(dk, []) if np.hypot(p[0] - x, p[1] - y) <= snap_px]
            fol[dk] = tuple(map(float, max(near, key=lambda p: p[2]))) if near else (x, y, -1.0)
    else: fol = follow(peaks, check["xy"], window)
    X, meta = [], []
    for dk, ball in sorted(fol.items()):
        if dk % every: continue
        f3 = (frames.get(dk - 1), frames.get(dk), frames.get(dk + 1))
        if any(f is None for f in f3): continue
        k = check["frame"] + dk
        for x, y, s, lab in frame_labels(peaks.get(dk, []), ball, n_neg_check if dk == 0 else n_neg_follow):
            X.append(BP.crop3(f3, x, y, size)); meta.append([f"{match}:{k}", "train", k, round(x, 1), round(y, 1), round(s, 4), lab, match, dk])
    return X, meta, fol


def sheet(X, meta, n=96, cols=12, zoom=3):
    """picture sheet of the ball crops (middle frame, zoomed) with their dk, to check by eye"""
    import cv2
    pos = [i for i, m in enumerate(meta) if m[6] == 1][:n]
    if not pos: return None
    s = X.shape[2] * zoom; rows = (len(pos) + cols - 1) // cols; img = np.zeros((rows * (s + 14), cols * s, 3), np.uint8)
    for j, i in enumerate(pos):
        r, c = divmod(j, cols); t = cv2.resize(X[i, 1], (s, s), interpolation=cv2.INTER_NEAREST)
        cv2.circle(t, (s // 2, s // 2), 5 * zoom, (0, 255, 255), 1)
        img[r * (s + 14):r * (s + 14) + s, c * s:(c + 1) * s] = t
        cv2.putText(img, f"{meta[i][8]:+d}", (c * s + 2, r * (s + 14) + s + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return img
