"""Ball auto-labels from track consistency (27 Sep). A single detection can be a shoe, a line crossing or a face in the
crowd; a detection that moves smoothly through 25+ consecutive frames, in a way a ball moves, almost never is.

Input: per-frame candidates {frame: [(x, y, conf), ...]} (any detector, at any stride). Detections are linked frame to
frame by nearest neighbour with a velocity guess; a track is kept when it is long, smooth (no jump), not static (clutter
in the trees never moves) and has a decent mean confidence. Every kept frame becomes a label; frames in the same seconds
as Daniel's clicks are excluded so the exam stays clean. A picture strip shows a sample for Daniel's eye."""
import numpy as np, cv2

def link(cands, fps=29.97, stride=1, max_step_px=60.0, min_len=20, max_gap=2, min_conf=0.12, min_travel_px=80.0, start_conf=0.25):
    """-> list of tracks, each [(frame, x, y, conf), ...]. max_step_px is per stride step."""
    frames = sorted(cands); tracks = []; active = []                          # active: [track, missed]
    for k in frames:
        dets = sorted(cands.get(k, []), key=lambda d: -d[2]); used = set(); nxt = []
        for tr, missed in active:
            f0, x0, y0, _ = tr[-1]; dt = (k - f0) / stride
            if dt > max_gap + 1: tracks.append(tr); continue
            vx, vy = (0.0, 0.0)
            if len(tr) >= 2: f1, x1, y1, _ = tr[-2]; d = max((f0 - f1) / stride, 1); vx, vy = (x0 - x1) / d, (y0 - y1) / d
            px, py = x0 + vx * dt, y0 + vy * dt; best = None
            for j, (x, y, c) in enumerate(dets):
                if j in used: continue
                r = np.hypot(x - px, y - py)
                if r < max_step_px * dt and (best is None or r < best[0]): best = (r, j)
            if best is not None: j = best[1]; used.add(j); tr.append((k, dets[j][0], dets[j][1], dets[j][2])); nxt.append([tr, 0])
            elif missed < max_gap: nxt.append([tr, missed + 1])
            else: tracks.append(tr)
        for j, (x, y, c) in enumerate(dets):
            if j not in used and c >= start_conf: nxt.append([[(k, x, y, c)], 0])
        active = nxt
    tracks += [tr for tr, _ in active]
    keep = []
    for tr in tracks:
        if len(tr) < min_len: continue
        P = np.array([[x, y] for _, x, y, _ in tr]); conf = np.mean([c for *_, c in tr])
        if conf < min_conf: continue
        if np.linalg.norm(P.max(0) - P.min(0)) < min_travel_px: continue          # static clutter
        steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
        if steps.max() > max_step_px * (max_gap + 1): continue
        keep.append(tr)
    return keep

def labels(tracks, exclude_seconds=(), fps=29.97, every=3, edge_trim=3):
    """(frame, x, y) labels from kept tracks: every `every`-th frame, track ends trimmed (linking is weakest there),
    never in a second Daniel clicked"""
    ex = {int(round(s)) for s in exclude_seconds}; out = []
    for tr in tracks:
        for i, (k, x, y, c) in enumerate(tr[edge_trim:len(tr) - edge_trim]):
            if i % every: continue
            if int(round(k / fps)) in ex: continue
            out.append((int(k), float(x), float(y), float(c)))
    out.sort(); return out

def strip(video_or_frames, labs, n=24, size=(640, 360), crop=160):
    """n labelled frames: the label ringed on a crop around it (so a 6 px ball is visible) plus the small full frame"""
    import random
    random.seed(0); pick = sorted(random.sample(labs, min(n, len(labs)))); tiles = []
    getter = video_or_frames if callable(video_or_frames) else (lambda k: video_or_frames.get(k))
    for k, x, y, c in pick:
        f = getter(k)
        if f is None: continue
        h, w = f.shape[:2]; x0, y0 = int(np.clip(x - crop / 2, 0, w - crop)), int(np.clip(y - crop / 2, 0, h - crop))
        cr = cv2.resize(f[y0:y0 + crop, x0:x0 + crop], (360, 360), interpolation=cv2.INTER_CUBIC)
        cv2.circle(cr, (int((x - x0) * 360 / crop), int((y - y0) * 360 / crop)), 18, (0, 255, 255), 2)
        sm = cv2.resize(f, size); cv2.circle(sm, (int(x * size[0] / w), int(y * size[1] / h)), 10, (0, 255, 255), 2)
        cv2.putText(sm, f"f{k} conf {c:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4); cv2.putText(sm, f"f{k} conf {c:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(np.hstack([sm, cr]))
    if not tiles: return None
    return np.vstack(tiles)

def on_pitch(labs, H, L=106.0, W=64.0, margin=2.0):
    """keep labels whose ground-plane position is on the pitch: trees, fences and signs above the far touchline map beyond
    it (27 Sep: a third of the first clip labels were in the trees because the camera pans and they glide like a ball).
    H: {frame: 3x3 pitch->pixels}"""
    from .calibration import to_m
    out = []
    for k, x, y, c in labs:
        Hk = H.get(k)
        if Hk is None: continue
        m = to_m(Hk, np.array([[x, y]], np.float32))[0]
        if -margin <= m[0] <= L + margin and -margin <= m[1] <= W + margin: out.append((k, x, y, c))
    return out

def track_on_pitch_share(tr, H, L=106.0, W=64.0, margin=2.0):
    labs = [(k, x, y, c) for k, x, y, c in tr if k in H]
    return len(on_pitch(labs, H, L, W, margin)) / max(1, len(labs))
