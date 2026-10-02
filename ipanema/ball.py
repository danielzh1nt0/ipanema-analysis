"""Ball: tiled detection (cached) -> candidates on the pitch -> trajectories in pitch space -> speed gate + confidence score -> picks."""
import os, pickle, numpy as np, json, cv2
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

def bridge(ball, fps, max_gap_s=1.0, bridged=None):
    """fill gaps up to max_gap_s by straight lines; `bridged` (a set) receives the filled frame numbers"""
    ks = sorted(ball); g = int(round(max_gap_s * fps))
    for a, b in zip(ks, ks[1:]):
        if 1 < b - a <= g:
            for k in range(a + 1, b):
                t = (k - a) / (b - a); ball[k] = [ball[a][0] + t * (ball[b][0] - ball[a][0]), ball[a][1] + t * (ball[b][1] - ball[a][1])]
                if bridged is not None: bridged.add(k)
    return ball

def _pan(H, i, j, pts):
    """pixels of frame i -> pixels of frame j (camera pan removed); fixed camera: unchanged"""
    if hasattr(H[i], "to_m") or hasattr(H[j], "to_m"): return np.asarray(pts, np.float32).reshape(-1, 2)
    G = H[j] @ np.linalg.inv(H[i])
    return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), G.astype(np.float32)).reshape(-1, 2)

def hold_still(ball, cands, H, fps, jump_px=110.0, hold_r=40.0, min_conf=0.3, look_s=1.0, min_share=0.6, max_gap_s=1.0, log=print):
    """S8/B10 (2 Oct): the picker teleports to another spot through a missed frame (pick -> miss -> far candidate costs 3.8,
    a direct jump 13). When it does, and the OLD spot keeps a candidate (conf >= min_conf within hold_r px, pan removed) in
    >= min_share of the next look_s seconds, the ball is still there: keep the old spot (its own candidates) until they
    stop, then return to the picker's path. Picks before the first frame are untouched."""
    ks = sorted(ball); out = dict(ball); held = 0; events = 0; i = 1
    while i < len(ks):
        a, b = ks[i - 1], ks[i]
        if b - a > max_gap_s * fps: i += 1; continue
        pa = _pan(H, a, b, [out[a]])[0]
        if np.hypot(pa[0] - out[b][0], pa[1] - out[b][1]) <= jump_px: i += 1; continue
        # candidate at the old spot in the following frames?
        look = [k for k in range(b, min(b + int(look_s * fps), len(H))) if cands.get(k)]
        hits = {}
        for k in look:
            p = _pan(H, a, k, [out[a]])[0]
            c = [c for c in cands[k] if c[2] >= min_conf and np.hypot(c[0] - p[0], c[1] - p[1]) <= hold_r]
            if c: hits[k] = max(c, key=lambda z: z[2])
        if not look or len(hits) / len(look) < min_share: i += 1; continue
        # hold: follow the old spot's candidates while they keep coming (gaps up to max_gap_s)
        events += 1; last = a; k = b
        while k < len(H) and k - last <= max_gap_s * fps:
            if cands.get(k):
                p = _pan(H, last, k, [out[last]])[0]
                c = [c for c in cands[k] if c[2] >= min_conf and np.hypot(c[0] - p[0], c[1] - p[1]) <= hold_r]
                if c:
                    best = max(c, key=lambda z: z[2]); out[k] = [best[0], best[1]]; last = k; held += 1
            k += 1
        # resume where the picker's own path is next, after the hold
        while i < len(ks) and ks[i] <= last: i += 1
    for k in list(out):
        if k not in ball and k not in cands: del out[k]
    log(f"ball hold-still: {events} teleports held, {held} frames moved to the still spot")
    return out

def still_spells(ball, H, fps, still_px=15.0, min_s=1.0, max_gap_s=0.3):
    """B10: stretches where the picked ball stays within still_px (camera pan removed) of where the stretch began for at
    least min_s seconds (missing picks up to max_gap_s allowed). Returns [(first frame, last frame, anchor xy)]."""
    ks = sorted(ball); out = []; i = 0; g = max_gap_s * fps
    while i < len(ks):
        a = ks[i]; last = a; j = i + 1
        while j < len(ks) and ks[j] - last <= g:
            p = _pan(H, a, ks[j], [ball[a]])[0]
            if np.hypot(p[0] - ball[ks[j]][0], p[1] - ball[ks[j]][1]) > still_px: break
            last = ks[j]; j += 1
        if last - a >= min_s * fps: out.append((a, last, list(ball[a]))); i = j
        else: i += 1
    return out

def still_prior(cands, ball, H, fps, per=None, still_px=15.0, min_s=1.0, hold_r=25.0, min_conf=0.05, boost=0.7,
                gap_s=0.3, player_m=2.0, discount=0.5, extend=True):
    """B10 (2 Oct): still-ball handling BEFORE the picker. After a first pick, find spells where the ball has not moved in
    the picture for > min_s (still_spells). For those frames (and, with extend, onwards while any candidate >= min_conf keeps
    showing within hold_r px of the still spot, gaps <= gap_s) the still spot gets a strong candidate (conf >= boost, at the
    spot's own candidate when there is one) and other candidates within player_m metres of a player have their conf
    multiplied by discount (the shoes/legs the finders flick to while the ball rests). Returns new candidates + spell list."""
    out = {k: list(v) for k, v in cands.items()}; n = len(H); spells = []
    covered = np.zeros(n, bool)
    for a, b, anchor in still_spells(ball, H, fps, still_px, min_s):
        if covered[a]: continue
        pos = np.float32(anchor); last = a; k = a; frames_ = []
        while k < n:
            p = _pan(H, a, k, [pos])[0] if k != a else pos
            near = [c for c in cands.get(k, []) if c[2] >= min_conf and np.hypot(c[0] - p[0], c[1] - p[1]) <= hold_r]
            if near or k <= b: frames_.append((k, near, p))
            if near: last = k
            if k > b and (not extend or k - last > gap_s * fps): break
            k += 1
        frames_ = [f for f in frames_ if f[0] <= max(b, last)]
        for k, near, p in frames_:
            covered[k] = True
            best = max(near, key=lambda z: z[2]) if near else (float(p[0]), float(p[1]), 0.0)
            rows = [c for c in out.get(k, []) if c not in near]
            if per is not None and per.get(k) and rows and discount < 1:
                pl = np.array([r[2] for r in per[k]]); m = to_m(H[k], np.float32([[c[0], c[1]] for c in rows]))
                rows = [(c[0], c[1], c[2] * discount) if np.linalg.norm(pl - mm, axis=1).min() <= player_m else c for c, mm in zip(rows, m)]
            out[k] = rows + [(best[0], best[1], max(best[2], boost))]
        spells.append((frames_[0][0], frames_[-1][0]))
    return out, spells

def pick_still(cands, H, L, W, per=None, fps=30.0, still_kw=None, log=print, **kw):
    """B10: pick, add the still-ball prior, pick again with the same picker settings."""
    first = pick_v2(cands, H, L, W, per=per, fps=fps, log=lambda *a: None, **kw)
    c2, spells = still_prior(cands, first, H, fps, per=per, **(still_kw or {}))
    log(f"ball still-prior: {len(spells)} still spells, {sum(b - a + 1 for a, b in spells)} frames")
    return pick_v2(c2, H, L, W, per=per, fps=fps, log=log, **kw)

def pick_confidence(ball, cands, r_px=12.0):
    """per frame: the strongest candidate score within r_px of the picked ball (0 when none, e.g. a bridged frame)"""
    out = {}
    for k, p in ball.items():
        cs = [c for x, y, c in cands.get(k, []) if np.hypot(x - p[0], y - p[1]) <= r_px]; out[k] = float(max(cs)) if cs else 0.0
    return out

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


def pick_global(cands, H, L, W, per=None, fps=30.0, margin=1.5, max_step_m=2.5, miss_cost=3.0, conf_w=1.5, near_w=2.5, min_conf=0.08, log=print):
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
        for r in C[i]:
            if r[2] < 0: continue
            grid.setdefault((i // win, round(r[0] * 2), round(r[1] * 2)), []).append(i)
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
                        tr = 0.4 * d + (6.0 if d > max_step_m * 3 else 0.0)
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


def recurring_spots(C, fps, L, W, margin=1.5, win_s=3.0, still_r=1.2, still_frac=0.5, merge_r=1.5, gap_s=30.0, min_visits=2):
    """B4b (1 Oct): off-pitch spots where something ball-like lies still in two or more separate visits (>= gap_s apart):
    a spare ball behind the goal, a cone, a bag. The match ball lies still off the pitch only once per restart, so it does
    not recur at the same spot. Players walking past do not matter here (the short-window clutter rule needs nobody near).
    C: per frame rows (mx, my, conf, ...). Returns a list of (x_m, y_m, [window starts in s])."""
    n = len(C); win = max(1, int(win_s * fps)); spots = []      # still spots per window: (x, y, window index)
    for w0 in range(0, n, win):
        pts = [(r[0], r[1], i) for i in range(w0, min(n, w0 + win)) for r in C[i]
               if r[2] >= 0 and not (-margin < r[0] < L + margin and -margin < r[1] < W + margin)]
        used = np.zeros(len(pts), bool); P = np.array([p[:2] for p in pts]) if pts else np.zeros((0, 2))
        while not used.all():
            best = None
            for j in np.flatnonzero(~used):
                m = (~used) & (np.linalg.norm(P - P[j], axis=1) <= still_r)
                fr = len({pts[q][2] for q in np.flatnonzero(m)})
                if best is None or fr > best[0]: best = (fr, m)
            fr, m = best
            if fr < still_frac * win: break
            spots.append((*np.median(P[m], axis=0), w0 // win)); used |= m
    groups = []                                                    # merge window spots that sit at the same place
    for x, y, k in spots:
        g = next((g for g in groups if np.hypot(g[0] - x, g[1] - y) <= merge_r), None)
        if g is None: groups.append([x, y, [k]])
        else: g[2].append(k); g[0] += (x - g[0]) / len(g[2]); g[1] += (y - g[1]) / len(g[2])
    out = []
    for x, y, ks in groups:
        t = sorted(k * win / fps for k in ks); visits = 1 + sum(1 for a, b in zip(t, t[1:]) if b - a >= gap_s)
        if visits >= min_visits: out.append((float(x), float(y), t))
    return out


def moving_ball(Mb, fps, L, W, margin=1.5, win_s=0.3, step_m=1.0, disp_m=0.6):
    """B4c: frames where a ball path (metres per frame, NaN = none) is on the pitch and moving smoothly: every frame in the
    last win_s seconds present and on the pitch, each step <= step_m (no jumps between guesses), and moved >= disp_m.
    margin < 0 = only count balls at least -margin metres inside the lines (a ball carried along the touchline by a
    ball boy during a stoppage is not play)."""
    n = len(Mb); k = max(1, int(round(win_s * fps)))
    with np.errstate(invalid="ignore"):
        on = (Mb[:, 0] > -margin) & (Mb[:, 0] < L + margin) & (Mb[:, 1] > -margin) & (Mb[:, 1] < W + margin)
    st = np.nan_to_num(np.r_[np.inf, np.linalg.norm(np.diff(Mb, axis=0), axis=1)], nan=np.inf)
    mv = np.zeros(n, bool)
    for j in range(k, n):
        if on[j - k:j + 1].all() and (st[j - k + 1:j + 1] <= step_m).all() and np.linalg.norm(Mb[j] - Mb[j - k]) >= disp_m: mv[j] = True
    return mv


def play_on_mask(moving, fps, play_s):
    """B4c: frame i is 'play on' if a moving ball was seen in the last play_s seconds (including frame i)."""
    n = len(moving); w = int(round(play_s * fps)); c = np.r_[0, np.cumsum(moving)]
    return np.array([c[i + 1] - c[max(0, i - w)] > 0 for i in range(n)], bool)


# ---- picker v2: movement measured in the picture with the camera pan removed; airborne balls kept ----
def pick_v2(cands, H, L, W, per=None, fps=30.0, margin=1.5, min_conf=0.08, conf_w=2.5, near_w=0.75, air_w=0.6,
            miss_cost=3.0, px_w=0.02, jump_px=110.0, jump_cost=6.0, gate_px=260.0, top_k=12, poss_cost=None, poss_px=(0.0, 0.0),
            poss_only_empty=False, recur_r=None, recur_play_s=None, recur_move_m=0.6, recur_inside_m=-1.5, ghost_s=None, max_ghosts=8, ghost_speed_px=40.0, reappear_cost=None, log=print):
    """1 Oct (tools/picktune.py, exact app inputs of both clips): conf_w 1.5->2.5, near_w 1.0->0.75 = trust the new finder
    more, the 'near a player' bonus less. AIK 26->30/39, SFK-BP 29->29/34, B4 key 279->284. (A first try, conf_w 4 / near_w 0.5 /
    miss_cost 5, was tuned on an older offline SFK-BP setup and lost one SFK-BP moment in the app.)
    poss_cost (B1, 28 Sep): also offer every player's feet as a 'ball with this player' candidate at this fixed cost,
    for moments when no finder sees the ball (at feet, in a crowd). Off (None) by default.
    ghost_s (S8, 2 Oct, tools/holdlab.py): missed frames carry the last position so a teleport through a miss costs a real
    jump. Off (None): on the exact clip inputs it removed 6 of 8 fake passes but lost 7 of 20 real ones and 3 AIK ball
    moments - a lost ball re-found far away looks like a teleport, and persistent clutter makes staying put cheap."""
    n = len(cands); C = []
    ppos = {i: np.array([r[2] for r in per[i]]) for i in range(n) if per and per.get(i)} if per is not None else {}
    inv = {}
    for i in range(n):
        rows = []
        if cands.get(i):
            top = [c for c in sorted(cands[i], key=lambda z: -z[2])[:top_k] if c[2] >= min_conf]
            if top:
                pts = np.float32([[x, y] for x, y, _ in top]).reshape(-1, 1, 2)
                m = to_m(H[i], pts.reshape(-1, 2))
                for (x, y, cf), (mx, my) in zip(top, m):
                    on = bool(-margin < mx < L + margin and -margin < my < W + margin)
                    near = float(np.linalg.norm(ppos[i] - np.array([mx, my]), axis=1).min() < 4.0) if (on and i in ppos) else 0.0
                    rows.append((float(mx), float(my), float(cf), float(x), float(y), near, on))
        if poss_cost is not None and per and per.get(i) and not (poss_only_empty and rows):
            for r in per[i]:
                if r[3] is None: continue
                mx, my = float(r[2][0]), float(r[2][1])
                rows.append((mx, my, -1.0, float(r[3][0]) + poss_px[0], float(r[3][1]) + poss_px[1], 1.0, True))
        C.append(rows)
    # B4b: recurring off-pitch spots (spare ball behind the goal), found before the short-window rule thins them out
    if recur_r:
        spots = recurring_spots(C, fps, L, W, margin=margin)
        play = None
        if spots and recur_play_s:
            # B4c: drop the spot only while play is on = a ball seen moving on the pitch in the last recur_play_s seconds
            # (first pass with the rule everywhere), so a ball taken from the post for a restart is followed again
            kw = dict(per=per, fps=fps, margin=margin, min_conf=min_conf, conf_w=conf_w, near_w=near_w, air_w=air_w,
                      miss_cost=miss_cost, px_w=px_w, jump_px=jump_px, jump_cost=jump_cost, gate_px=gate_px, top_k=top_k,
                      poss_cost=poss_cost, poss_px=poss_px, poss_only_empty=poss_only_empty, ghost_s=ghost_s, max_ghosts=max_ghosts, ghost_speed_px=ghost_speed_px, reappear_cost=reappear_cost)
            first = pick_v2(cands, H, L, W, recur_r=recur_r, log=lambda *a: None, **kw)
            Mb = np.full((n, 2), np.nan)
            for i, p in first.items(): Mb[i] = to_m(H[i], np.float32([p]))[0]
            play = play_on_mask(moving_ball(Mb, fps, L, W, margin=-recur_inside_m, disp_m=recur_move_m), fps, recur_play_s)
            log(f"ball v2: play on (ball moving on the pitch within {recur_play_s} s) on {play.mean():.0%} of frames")
        if spots:
            S = np.array([s[:2] for s in spots]); rd = 0
            for i in range(n):
                if play is not None and not play[i]: continue
                keep = [r for r in C[i] if r[2] < 0 or r[6] or np.linalg.norm(S - (r[0], r[1]), axis=1).min() > recur_r]
                rd += len(C[i]) - len(keep); C[i] = keep
            log(f"ball v2: {len(spots)} recurring off-pitch spots ({', '.join(f'{x:.0f},{y:.0f}' for x, y, _ in spots)}), {rd} candidates dropped")
    # static clutter (fence signs, cones, marks): same projected position for seconds with nobody near it
    win = int(3 * fps); grid = {}
    def cell(i, r): return (i // win, round(r[0] * 2), round(r[1] * 2))   # S8 (2 Oct): pan-corrected pixel cells tried instead - worse on all 3 keys (drops real still balls)
    for i in range(n):
        for r in C[i]:
            if r[2] < 0: continue
            grid.setdefault(cell(i, r), set()).add(i)
    dropped = 0
    for i in range(n):
        keep = []
        for r in C[i]:
            if r[2] < 0: keep.append(r); continue
            if r[5] == 0.0 and len(grid.get(cell(i, r), ())) > 0.6 * win: dropped += 1; continue
            keep.append(r)
        C[i] = keep
    if ghost_s:
        ball = _viterbi_ghost(C, H, n, fps, conf_w, near_w, air_w, miss_cost, px_w, jump_px, jump_cost, gate_px, poss_cost, ghost_s, max_ghosts, ghost_speed_px, jump_cost if reappear_cost is None else reappear_cost)
        log(f"ball v2 (ghosts {ghost_s} s): {len(ball)}/{n} frames on the path, {dropped} static-clutter candidates dropped")
        return ball
    INF = 1e18; back = []; prev_cost = None
    for i in range(n):
        rows = C[i]; k = len(rows)
        emit = np.array([poss_cost if r[2] < 0 else conf_w * (1 - r[2]) + (near_w * (1 - r[5]) if r[6] else air_w) for r in rows] + [miss_cost])
        if prev_cost is None:
            cur = emit.copy(); bk = np.full(k + 1, -1, int)
        else:
            prows = C[i - 1]; pk = len(prows)
            if k and pk:
                if hasattr(H[i], "to_m"): q = np.float32([[r[3], r[4]] for r in rows])     # fixed camera: nothing to remove
                else:
                    G = H[i - 1] @ np.linalg.inv(H[i])                   # frame i pixels -> frame i-1 pixels (camera pan removed)
                    q = cv2.perspectiveTransform(np.float32([[r[3], r[4]] for r in rows]).reshape(-1, 1, 2), G.astype(np.float32)).reshape(-1, 2)
                p = np.float32([[r[3], r[4]] for r in prows])
                d = np.linalg.norm(q[:, None, :] - p[None, :, :], axis=2)   # (k, pk) pixels
                tr = px_w * d + np.where(d > jump_px, jump_cost, 0.0) + np.where(d > gate_px, 1e6, 0.0)
            else:
                tr = np.zeros((k, pk))
            full = np.full((k + 1, pk + 1), 0.8); full[:k, :pk] = tr; full[k, pk] = 0.0
            tot = prev_cost[None, :] + full
            bk = np.argmin(tot, axis=1); cur = tot[np.arange(k + 1), bk] + emit
        back.append(bk); prev_cost = cur
    ball = {}; s = int(np.argmin(prev_cost))
    for i in range(n - 1, -1, -1):
        if s < len(C[i]): r = C[i][s]; ball[i] = [r[3], r[4]]
        s = back[i][s]
        if s < 0: break
    log(f"ball v2: {len(ball)}/{n} frames on the path, {dropped} static-clutter candidates dropped")
    return ball


def _viterbi_ghost(C, H, n, fps, conf_w, near_w, air_w, miss_cost, px_w, jump_px, jump_cost, gate_px, poss_cost, ghost_s, max_ghosts, ghost_speed_px=40.0, reappear_cost=6.0):
    """S8 (2 Oct): the plain path could teleport through a missed frame (pick -> miss 3 -> far candidate 0.8 = 3.8, a
    direct jump 13). Here a missed frame keeps the last position as a 'ghost' state (costs miss_cost like a miss, carries
    the pixel position pan-corrected, lives ghost_s seconds, at most max_ghosts per frame); a candidate after a ghost pays the
    distance like after a seen ball, and a candidate after the bare 'miss' state (ball gone for > ghost_s) pays jump_cost.
    States per frame: candidates, ghosts, miss (last). Backtracking yields picks only on candidate states."""
    INF = 1e18; back = []; pos_hist = []; kind_hist = []; prev_cost = None; prev_pos = None; prev_kind = None; prev_age = None
    for i in range(n):
        rows = C[i]; k = len(rows)
        rpos = np.float32([[r[3], r[4]] for r in rows]).reshape(-1, 2)
        remit = [poss_cost if r[2] < 0 else conf_w * (1 - r[2]) + (near_w * (1 - r[5]) if r[6] else air_w) for r in rows]
        if prev_cost is None:
            cost = np.array(remit + [miss_cost]); bk = np.full(k + 1, -1, int); pos = rpos; kind = ["c"] * k + ["m"]; age = [0] * (k + 1)
            gpos = np.zeros((0, 2), np.float32)
        else:
            pk = len(prev_cost); pm = pk - 1                       # previous miss index
            # previous positions carried into this frame's pixels
            carried = _pan(H, i - 1, i, prev_pos) if len(prev_pos) else np.zeros((0, 2), np.float32)
            # ghosts: from previous candidate/ghost states still young enough; keep the cheapest, dedup 10 px, cap
            cand_g = [j for j in range(pm) if prev_age[j] + 1 <= int(ghost_s * fps)]
            cand_g.sort(key=lambda j: prev_cost[j]); gsrc = []
            for j in cand_g:
                if all(np.hypot(*(carried[j] - carried[q])) > 10 for q in gsrc): gsrc.append(j)
                if len(gsrc) >= max_ghosts: break
            gpos = carried[gsrc] if gsrc else np.zeros((0, 2), np.float32); g = len(gsrc)
            m = k + g + 1; cost = np.full(m, INF); bk = np.full(m, -1, int)
            if k:
                d = np.linalg.norm(rpos[:, None, :] - carried[None, :, :], axis=2) if pm else np.zeros((k, 0))
                # a ghost of age a frames may have flown ghost_speed_px * a further (a real ball lost in flight); a teleport cannot
                allow = jump_px + ghost_speed_px * np.array(prev_age[:pm], float)[None, :]
                tr = px_w * np.maximum(0.0, d - ghost_speed_px * np.array(prev_age[:pm], float)[None, :]) + np.where(d > allow, jump_cost, 0.0) + np.where(d > gate_px + ghost_speed_px * np.array(prev_age[:pm], float)[None, :], 1e6, 0.0)
                full = np.full((k, pk), 0.0); full[:, :pm] = tr; full[:, pm] = reappear_cost  # from miss (ball gone > ghost_s): a reappearance
                tot = prev_cost[None, :] + full; bkc = np.argmin(tot, axis=1)
                cost[:k] = tot[np.arange(k), bkc] + np.array(remit); bk[:k] = bkc
            for t, j in enumerate(gsrc): cost[k + t] = prev_cost[j] + miss_cost; bk[k + t] = j
            tot_m = prev_cost + np.where(np.arange(pk) == pm, 0.0, 0.8); bk[m - 1] = int(np.argmin(tot_m)); cost[m - 1] = tot_m[bk[m - 1]] + miss_cost
            pos = np.vstack([rpos, gpos]) if (k or g) else np.zeros((0, 2), np.float32); kind = ["c"] * k + ["g"] * g + ["m"]
            age = [0] * k + [prev_age[j] + 1 for j in gsrc] + [0]
        back.append(bk); pos_hist.append(pos); kind_hist.append(kind); prev_cost, prev_pos, prev_kind, prev_age = cost, pos, kind, age
    ball = {}; s = int(np.argmin(prev_cost))
    for i in range(n - 1, -1, -1):
        if kind_hist[i][s] == "c": ball[i] = [float(pos_hist[i][s][0]), float(pos_hist[i][s][1])]
        s = back[i][s]
        if s < 0: break
    return ball


def fuse_candidates(clicks, wasb, wy=1.5, wb=1.0, bonus=0.8, px=12):
    """one guess list per frame from both detectors (27 Sep, graded on the clip's 34 checked moments: 26/34 with the v2
    picker vs 21 WASB-only / 20 click-only). A guess both detectors agree on (within px) gets both scores plus a bonus."""
    out = {}
    for k in set(clicks) | set(wasb):
        cands = [[x, y, wy * c] for x, y, c in clicks.get(k, [])]
        for x, y, s in wasb.get(k, []):
            j = next((i for i, q in enumerate(cands) if np.hypot(q[0] - x, q[1] - y) <= px), None)
            if j is not None: cands[j][2] += wb * s + bonus
            else: cands.append([x, y, wb * s])
        out[k] = [(x, y, min(0.99, sc / 2.0)) for x, y, sc in cands]
    return out
