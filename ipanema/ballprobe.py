"""B2 (28 Sep): do the ball finders see the ball WEAKLY at the moments we miss? For each checked moment, run every finder
variant at a very low cut-off and record whether the true ball is among the guesses, at what rank and score.
Pure logic here (tested with stand-in finders); the GPU wrapper is modal_app.ball_probe."""
import numpy as np

HIT_PX = 30.0


def flip_back(dets, width):
    """guesses found on a mirrored frame -> original frame coordinates"""
    return [(width - 1 - x, y, s) for x, y, s in dets]


def tiles_2x2(shape, overlap=0.15):
    h, w = shape[:2]; tw, th = int(w / 2 * (1 + overlap)), int(h / 2 * (1 + overlap))
    return [(x0, y0, x0 + tw, y0 + th) for y0 in (0, h - th) for x0 in (0, w - tw)]


def detect_tiled(detect, frame, overlap=0.15, merge_px=12.0):
    """run `detect(img) -> [(x, y, s)]` on 4 overlapping quarter tiles (each shown to the model at full size = ~2x zoom)"""
    out = []
    for x0, y0, x1, y1 in tiles_2x2(frame.shape, overlap):
        out += [(x + x0, y + y0, s) for x, y, s in detect(frame[y0:y1, x0:x1])]
    return merge(out, merge_px)


def merge(dets, px=12.0):
    keep = []
    for d in sorted(dets, key=lambda z: -z[2]):
        if all(np.hypot(d[0] - k[0], d[1] - k[1]) > px for k in keep): keep.append(d)
    return keep


def grade(dets, truth, hit_px=HIT_PX):
    """rank (0 = top), score and distance of the true ball among the guesses; truth None = no ball in the frame"""
    dets = sorted(dets, key=lambda z: -z[2])
    if truth is None: return {"has_ball": False, "n": len(dets), "top": round(dets[0][2], 4) if dets else None}
    d = [float(np.hypot(x - truth[0], y - truth[1])) for x, y, _ in dets]
    hit = next((i for i, dd in enumerate(d) if dd <= hit_px), None)
    return {"has_ball": True, "n": len(dets), "rank": hit, "score": round(dets[hit][2], 4) if hit is not None else None,
            "nearest_px": round(min(d), 1) if d else None}


def summarise(rows, variants, normal_cut=0.05):
    """rows: [{"id", "group", "truth", "<variant>": grade}] -> per variant: ball among guesses at all, at >= the normal
    cut-off, only below it; plus 'any variant' and the moments still unseen by everything"""
    wb = [r for r in rows if r["truth"] is not None]; s = {"ball_moments": len(wb), "variants": {}}
    for v in variants:
        g = [r[v] for r in wb if "rank" in r.get(v, {})]
        s["variants"][v] = {"seen_at_all": sum(1 for x in g if x["rank"] is not None),
                            "seen_at_normal_cut": sum(1 for x in g if x["score"] is not None and x["score"] >= normal_cut), "graded": len(g),
                            "seen_only_below_cut": sum(1 for x in g if x["score"] is not None and x["score"] < normal_cut),
                            "top_guess": sum(1 for x in g if x["rank"] == 0)}
    seen_any = [r for r in wb if any(r.get(v, {}).get("rank") is not None for v in variants)]
    s["seen_by_any_variant"] = len(seen_any)
    s["never_seen"] = [r["id"] for r in wb if r not in seen_any]
    return s


def local_peaks(hm, thr=0.01, max_n=30, nms=5):
    """local maxima of a heatmap above thr (low cut-offs merge blobs, so centroids of blobs would drift)"""
    import cv2
    hm = np.asarray(hm, np.float32)
    if hm.max() <= thr: return []
    mx = cv2.dilate(hm, np.ones((2 * nms + 1, 2 * nms + 1), np.uint8))
    ys, xs = np.where((hm >= mx) & (hm > thr))
    pk = sorted(((float(x), float(y), float(hm[y, x])) for x, y in zip(xs, ys)), key=lambda z: -z[2])
    return merge(pk, nms)[:max_n]


def run(moments, read3, variants, log=print, keep=30):
    """moments: [{"id", "frame", "truth": (x, y) | None, "group"}]; read3(frame) -> (prev, cur, next) images or None;
    variants: {name: fn(frames3) -> [(x, y, score)]}. Returns rows with each variant's grade + its top guesses."""
    rows = []
    for i, m in enumerate(moments):
        f3 = read3(m["frame"])
        if f3 is None: log(f"no frame {m['frame']}"); continue
        r = {"id": m["id"], "frame": m["frame"], "group": m.get("group"), "truth": m["truth"], "guesses": {}}
        for name, fn in variants.items():
            try: d = merge(fn(f3))
            except Exception as e: r[name] = {"error": repr(e)[:200]}; continue
            r[name] = grade(d, m["truth"]); r["guesses"][name] = [[round(x, 1), round(y, 1), round(s, 4)] for x, y, s in d[:keep]]
        rows.append(r)
        if (i + 1) % 20 == 0: log(f"{i + 1}/{len(moments)} moments")
    return rows


def crop3(f3, x, y, size=32):
    """size x size crop centred on (x, y) from each of the 3 frames, zero-padded at the edges -> (3, size, size, 3) uint8"""
    h, w = f3[1].shape[:2]; r = size // 2; out = np.zeros((3, size, size, 3), np.uint8)
    x0, y0 = int(round(x)) - r, int(round(y)) - r
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0 + size), min(h, y0 + size)
    if sx1 <= sx0 or sy1 <= sy0: return out
    for i, f in enumerate(f3): out[i, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = f[sy0:sy1, sx0:sx1]
    return out


def label_guesses(cands, truth, pos_px=10.0, neg_px=30.0, n_neg=12):
    """guesses -> [(x, y, score, label)]: 1 = the ball (within pos_px of the click), 0 = not the ball (> neg_px, or no
    ball in the frame), in-between left out. Keeps every positive and the n_neg strongest negatives. If no guess is on
    the ball, the click itself is added as a positive (score -1), so the scorer still sees what the ball looks like."""
    out = []; negs = []
    for x, y, s in sorted(cands, key=lambda z: -z[2]):
        if truth is None: negs.append((x, y, s, 0)); continue
        d = float(np.hypot(x - truth[0], y - truth[1]))
        if d <= pos_px: out.append((x, y, s, 1))
        elif d > neg_px: negs.append((x, y, s, 0))
    if truth is not None and not out: out.append((float(truth[0]), float(truth[1]), -1.0, 1))
    return out + negs[:n_neg]
