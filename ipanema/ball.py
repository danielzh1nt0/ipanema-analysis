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

def check(ball, cands, gt_path, hit_px=30, log=print):
    if not gt_path or not os.path.exists(gt_path): return None
    gt = {int(k): v for k, v in json.load(open(gt_path)).items()}
    tot = ok = wrong = ceil = 0
    for i, g in gt.items():
        if g is None: continue
        tot += 1
        if cands.get(i) and min(np.hypot(x - g[0], y - g[1]) for x, y, _ in cands[i]) <= hit_px: ceil += 1
        if i in ball:
            if np.hypot(ball[i][0] - g[0], ball[i][1] - g[1]) <= hit_px: ok += 1
            else: wrong += 1
    log(f"ball check: {ok}/{tot} correct, {wrong} wrong, {tot-ok-wrong} no pick (ceiling {ceil}/{tot})"); return {"correct": ok, "total": tot, "wrong": wrong, "ceiling": ceil}
