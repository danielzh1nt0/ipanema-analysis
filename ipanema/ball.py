"""Ball: tiled detection (cached) -> candidates on the pitch -> trajectories in pitch space -> speed gate + confidence score -> picks."""
import os, pickle, numpy as np, json
from .video import frames
from .calibration import to_m

def candidates(video, weights_ball, cache, conf=0.05, imgsz=1920, tiles=(3, 2), overlap=0.2, log=print):
    from ultralytics import YOLO
    out = {}; partial = cache + ".partial"
    if os.path.exists(cache): return pickle.load(open(cache, "rb"))
    if os.path.exists(partial): out = pickle.load(open(partial, "rb")); log(f"  ball: resuming from frame {len(out)}")
    model = YOLO(weights_ball)
    def run(img, ox, oy, acc):
        r = model(img, conf=conf, imgsz=imgsz, verbose=False)[0]
        for (x1, y1, x2, y2), cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()): acc.append(((x1 + x2) / 2 + ox, (y1 + y2) / 2 + oy, float(cf)))
    for k, f in frames(video):
        if k in out: continue
        Hh, Ww = f.shape[:2]; acc = []; run(f, 0, 0, acc); tw, th = Ww / tiles[0], Hh / tiles[1]
        for r_ in (range(tiles[1]) if not any(cf >= 0.3 for _, _, cf in acc) else []):   # tiles only when the full frame is unsure
            for c_ in range(tiles[0]):
                x0 = int(max(0, c_ * tw - tw * overlap)); y0 = int(max(0, r_ * th - th * overlap)); x1 = int(min(Ww, (c_ + 1) * tw + tw * overlap)); y1 = int(min(Hh, (r_ + 1) * th + th * overlap))
                run(f[y0:y1, x0:x1], x0, y0, acc)
        merged = []
        for x, y, cf in sorted(acc, key=lambda z: -z[2]):
            if all(np.hypot(x - mx, y - my) > 12 for mx, my, _ in merged): merged.append((x, y, cf))
        out[k] = merged
        if k % 500 == 0:
            log(f"  ball frame {k}: {len(merged)} candidates"); pickle.dump(out, open(partial, "wb"))
    pickle.dump(out, open(cache, "wb"))
    if os.path.exists(partial): os.remove(partial)
    return out

def pick(cands, H, L, W, margin=1.5, link_r=45, max_gap=6, min_speed=0.5, min_score=4.0, seed_conf=0.15, per=None, log=print):
    n = len(cands); S = {}
    for i in range(n):
        S[i] = []
        if not cands[i]: continue
        m = to_m(H[i], [[x, y] for x, y, _ in cands[i]])
        for (x, y, cf), (mx, my) in zip(cands[i], m):
            if -margin < mx < L + margin and -margin < my < W + margin: S[i].append((mx * 10, my * 10, cf, x, y))
    # a real ball is near a player most of the time: precompute player positions (metres) per frame for scoring
    ppos = {i: np.array([r[2] for r in per[i]]) for i in range(n) if per and per.get(i)} if per is not None else {}
    tracks, active = [], []
    for i in range(n):
        used = set()
        for tr in active:
            lf, lx, ly = tr[-1][0], tr[-1][1], tr[-1][2]
            if i - lf > max_gap: continue
            best, bd = None, None
            for k, (sx, sy, cf, x, y) in enumerate(S[i]):
                if k in used: continue
                d = np.hypot(sx - lx, sy - ly) / (i - lf)
                if d <= link_r and (bd is None or d < bd): best, bd = k, d
            if best is not None: sx, sy, cf, x, y = S[i][best]; tr.append((i, sx, sy, cf, x, y)); used.add(best)
        for k, (sx, sy, cf, x, y) in enumerate(S[i]):
            if k not in used and cf >= seed_conf: t = [(i, sx, sy, cf, x, y)]; tracks.append(t); active.append(t)   # only confident candidates start a track
        active = [t for t in active if i - t[-1][0] <= max_gap]
    def speed(tr):
        P = np.array([[t[1], t[2]] for t in tr]); F = np.array([t[0] for t in tr]); return np.median(np.linalg.norm(np.diff(P, axis=0), axis=1) / np.diff(F))
    def score(tr):
        if len(tr) < 5 or speed(tr) < min_speed: return 0.0
        P = np.array([[t[1], t[2]] for t in tr]); F = np.array([t[0] for t in tr]); C = np.array([t[3] for t in tr])
        v = np.diff(P, axis=0) / np.diff(F)[:, None]; acc = np.linalg.norm(np.diff(v, axis=0), axis=1).mean() if len(v) > 1 else 0
        # long, smooth, consistently detected tracks win; absolute confidence matters less (a small ball is always low-confidence)
        near = 1.0
        if ppos:
            hits = [np.linalg.norm(ppos[t[0]] - np.array([t[1], t[2]]) / 10.0, axis=1).min() < 4.0 for t in tr[::3] if t[0] in ppos]
            near = 0.5 + float(np.mean(hits)) if hits else 1.0
        return len(tr) * (0.3 + C.mean()) * near / (1 + acc / 5)
    scored = sorted(((score(t), t) for t in tracks), key=lambda z: -z[0]); ball = {}
    for s, tr in scored:
        if s < min_score: break
        for t in tr: ball.setdefault(t[0], [float(t[4]), float(t[5])])
    top = [(round(s, 1), len(t), round(float(np.mean([q[3] for q in t])), 2), round(float(speed(t)), 2)) for s, t in scored[:5]]
    log(f"ball: {sum(len(v) for v in cands.values())/max(1,n):.1f} candidates/frame, {len(tracks)} trajectories, top (score,len,conf,speed) {top}, picks {len(ball)}/{n}")
    return ball

def bridge(ball, fps, max_gap_s=1.0):
    ks = sorted(ball); g = int(round(max_gap_s * fps))
    for a, b in zip(ks, ks[1:]):
        if 1 < b - a <= g:
            for k in range(a + 1, b):
                t = (k - a) / (b - a); ball[k] = [ball[a][0] + t * (ball[b][0] - ball[a][0]), ball[a][1] + t * (ball[b][1] - ball[a][1])]
    return ball

def check(ball, cands, gt_path, hit_px=30, log=print, video=None, debug_dir=None):
    if not gt_path or not os.path.exists(gt_path): return None
    gt = {int(k): v for k, v in json.load(open(gt_path)).items()}
    if video and debug_dir:
        # diagnostic tiles: your click (green), our pick (red), candidates (yellow, size ~ confidence), 240 px around the truth
        try:
            import cv2; os.makedirs(debug_dir, exist_ok=True); cap = cv2.VideoCapture(video); tiles = []
            for i, g in sorted(gt.items()):
                if g is None: continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
                if not ok: continue
                for x, y, cf in cands.get(i, []): cv2.circle(f, (int(x), int(y)), int(6 + 20 * cf), (0, 255, 255), 1)
                if i in ball: cv2.drawMarker(f, (int(ball[i][0]), int(ball[i][1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
                cv2.circle(f, (int(g[0]), int(g[1])), 14, (0, 255, 0), 2)
                x0, y0 = int(max(0, g[0] - 160)), int(max(0, g[1] - 90)); t = f[y0:y0 + 180, x0:x0 + 320]
                if t.shape[:2] != (180, 320): t = cv2.copyMakeBorder(t, 0, 180 - t.shape[0], 0, 320 - t.shape[1], cv2.BORDER_CONSTANT)
                d = np.hypot(ball[i][0] - g[0], ball[i][1] - g[1]) if i in ball else None
                cv2.putText(t, f"f{i} " + ("miss" if d is None else f"{d:.0f}px"), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1); tiles.append(t)
            cap.release()
            while len(tiles) % 4: tiles.append(np.zeros((180, 320, 3), np.uint8))
            sheet = np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)])
            cv2.imwrite(os.path.join(debug_dir, "ballcheck.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        except Exception as e: log(f"  ball check tiles failed: {e!r}")
    tot = ok = wrong = ceil = 0
    for i, g in gt.items():
        if g is None: continue
        tot += 1
        if cands.get(i) and min(np.hypot(x - g[0], y - g[1]) for x, y, _ in cands[i]) <= hit_px: ceil += 1
        if i in ball:
            if np.hypot(ball[i][0] - g[0], ball[i][1] - g[1]) <= hit_px: ok += 1
            else: wrong += 1
    log(f"ball check: {ok}/{tot} correct, {wrong} wrong, {tot-ok-wrong} no pick (ceiling {ceil}/{tot})"); return {"correct": ok, "total": tot, "wrong": wrong, "ceiling": ceil}


def pick_global(cands, H, L, W, per=None, fps=30.0, margin=1.5, max_step_m=2.5, miss_cost=3.5, conf_w=2.5, near_w=2.5, min_conf=0.10, log=print):
    """ONE ball path through the whole clip: dynamic programming over per-frame candidates plus a 'no ball' state.
    Staying with a consistent, confident, player-adjacent path is cheap; jumping is expensive. Returns {frame: [x_px, y_px]}."""
    n = len(cands); C = []      # per frame: list of (mx, my, conf, x, y, near)
    ppos = {i: np.array([r[2] for r in per[i]]) for i in range(n) if per and per.get(i)} if per is not None else {}
    for i in range(n):
        rows = []
        if cands.get(i):
            top = sorted(cands[i], key=lambda z: -z[2])[:12]
            m = to_m(H[i], [[x, y] for x, y, _ in top])
            for (x, y, cf), (mx, my) in zip(top, m):
                if cf < min_conf or not (-margin < mx < L + margin and -margin < my < W + margin): continue
                near = float(np.linalg.norm(ppos[i] - np.array([mx, my]), axis=1).min() < 4.0) if i in ppos else 0.5
                rows.append((float(mx), float(my), float(cf), float(x), float(y), near))
        C.append(rows)
    # static clutter: a candidate that sits at the same pitch position for seconds with nobody near it is a cone / spare ball / mark, not the ball
    win = int(3 * fps); grid = {}
    for i in range(n):
        for r in C[i]: grid.setdefault((i // win, round(r[0] * 2), round(r[1] * 2)), []).append(i)
    dropped = 0
    for i in range(n):
        keep = []
        for r in C[i]:
            key = (i // win, round(r[0] * 2), round(r[1] * 2)); frames_here = len(set(grid.get(key, [])))
            if r[5] == 0.0 and frames_here > 0.6 * win: dropped += 1; continue
            keep.append(r)
        C[i] = keep
    if dropped: log(f"ball: dropped {dropped} static-clutter candidates")
    INF = 1e18; cost = []; back = []; last_pos = {}      # state index len(rows) = "no ball"
    prev_cost = None
    for i in range(n):
        rows = C[i]; k = len(rows); cur = np.full(k + 1, INF); bk = np.full(k + 1, -1, int)
        emit = np.array([conf_w * (1 - r[2]) + near_w * (1 - r[5]) for r in rows] + [miss_cost])
        if prev_cost is None:
            cur = emit; bk[:] = -1
        else:
            prows = C[i - 1]; pk = len(prows)
            for s in range(k + 1):
                best, arg = INF, -1
                for ps in range(pk + 1):
                    if s < k and ps < pk:
                        d = np.hypot(rows[s][0] - prows[ps][0], rows[s][1] - prows[ps][1])
                        tr = 0.25 * d + (8.0 if d > max_step_m * 3 else 0.0)      # smoother paths, heavier jump penalty
                    elif s < k or ps < pk: tr = 0.8       # entering / leaving "no ball"
                    else: tr = 0.0
                    c = prev_cost[ps] + tr
                    if c < best: best, arg = c, ps
                cur[s] = best + emit[s]; bk[s] = arg
        cost.append(cur); back.append(bk); prev_cost = cur
    ball = {}; s = int(np.argmin(cost[-1]))
    for i in range(n - 1, -1, -1):
        if s < len(C[i]): r = C[i][s]; ball[i] = [r[3], r[4]]
        s = back[i][s]
        if s < 0: break
    log(f"ball (global path): {len(ball)}/{n} frames on the chosen path")
    return ball
