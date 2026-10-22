"""N0-22e: how wrong are the team shape numbers (stats.json teams[].block_length_median_m, block_width_median_m,
def_line_height_median_m; the app's line-height timeline / block cards read the same per-frame shapes) when the camera
only shows part of the pitch, like our Veo follow-cam?

Truth = Metrica's free pro tracking (all 20 outfield players, 25 fps). Each Metrica frame gets a REAL camera view from
our app clips: the pitch area our camera showed at that moment (from the clip's calibration), slid along the pitch so
its centre sits on Metrica's ball. Only outfield players inside that view are 'seen'. Then the shape numbers are
computed as analytics.stats does (>= 5 seen outfield players) and compared with the same numbers from all players.

Rules compared (all on the same frames):
  all      - every outfield player (the truth)
  seen5    - today's rule: seen players, frames with >= 5 seen
  seenN    - frames with >= N seen (N = 7, 8, 9)
  cover    - today's numbers, but a frame counts only when the camera also shows the pitch BEYOND the team's
             outermost seen players (behind the deepest one for the line height, beyond both ends for the length,
             beyond both sides for the width) by at least --margin m over most of the pitch width: then a hidden
             player out there would have been in view, so the seen extreme is very likely the real one.
Usage: python tools/n0_22e_shape_view.py <metrica data dir> <picker_inputs.pkl>... [--out f.json] [--margin 5]
       python tools/n0_22e_shape_view.py --dry   (stand-ins, for the test suite)"""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import analytics as AN

GX, GY = 1.0, 2.0          # view grid (m)


def view_masks(path, every=15):
    """our camera's view per sampled frame: boolean grid over the clip's pitch (x every GX m, y every GY m), plus the
    view's centre x. Pitch points are projected into the picture with the frame's H (metres -> pixels)."""
    d = pickle.load(open(path, "rb")); H, L, W = d["H"], float(d["L"]), float(d["W"]); n = len(d["per"])
    xs = np.arange(0, L + 1e-6, GX); ys = np.arange(0, W + 1e-6, GY); X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), np.ones(X.size)])
    out = []
    for k in range(0, n, every):
        if k not in H or H[k] is None or hasattr(H[k], "to_m"): continue
        v = np.asarray(H[k], float) @ P; z = v[2]; u, w = v[0] / np.where(np.abs(z) < 1e-9, 1e-9, z), v[1] / np.where(np.abs(z) < 1e-9, 1e-9, z)
        m = ((z > 0) & (u >= 0) & (u < 1920) & (w >= 0) & (w < 1080)).reshape(X.shape)
        if m.sum() < 20: continue
        out.append((m, float(X[m].mean()) / L))
    return out, L, W


def place(mask, cx, bx, L, W, Lm, Wm):
    """a view mask from a clip (pitch L x W) slid along x so its centre is at Metrica's ball; returns a function
    (x, y) in Metrica metres -> in view?"""
    sx = (bx / Lm) - cx                                     # shift in pitch fractions
    nx, ny = mask.shape
    def inside(x, y):
        i = np.round(((np.asarray(x) / Lm - sx) * L) / GX).astype(int); j = np.round((np.asarray(y) / Wm * W) / GY).astype(int)
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        r = np.zeros(np.shape(x), bool); r[ok] = mask[i[ok], j[ok]]
        return r
    return inside


def shape(pts, attack_right, L):
    xs = np.sort(pts[:, 0])
    return xs[-1] - xs[0], pts[:, 1].max() - pts[:, 1].min(), (xs[0] if attack_right else L - xs[-1])


def covered(inside, pts, attack_right, L, W, margin, share=0.7):
    """does the view show the pitch beyond the seen extremes (behind the deepest, ahead of the highest, outside both
    sides) by `margin` m over >= share of the width / length? -> (length ok, width ok, line ok)"""
    ys = np.linspace(1, W - 1, 15); xs_ = np.linspace(1, L - 1, 25)
    lo, hi = pts[:, 0].min(), pts[:, 0].max(); y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    def band_x(x): x = np.clip(x, 0, L); return inside(np.full_like(ys, x), ys).mean() >= share
    def band_y(y): y = np.clip(y, 0, W); xx = xs_[(xs_ >= lo - 5) & (xs_ <= hi + 5)]; return len(xx) == 0 or inside(xx, np.full_like(xx, y)).mean() >= share
    beh = band_x(lo - margin) if attack_right else band_x(hi + margin)          # behind the deepest player (towards his own goal)
    ahead = band_x(hi + margin) if attack_right else band_x(lo - margin)
    side = band_y(y0 - margin) and band_y(y1 + margin)
    return beh and ahead, side, beh


def run(frames, views, Lv, Wv, Lm=105.0, Wm=68.0, margin=5.0, seed=0, step=12, attack_right=None):
    """frames: {frame: (period, [(id, x, y)], ball or None)} (Metrica layout, metres). Teams: id < 200 home, else away.
    Keepers = per team and period the player with the most extreme mean x. -> {team: {rule: {length, width, line, n}}}"""
    rng = np.random.default_rng(seed); ks = sorted(frames)[::step]
    # keepers and directions per period
    meanx = {}
    for k in ks:
        p, ps, _ = frames[k]
        for i, x, y in ps: meanx.setdefault((p, i >= 200), {}).setdefault(i, []).append(x)
    gk, right = {}, {}
    for (p, away), d in meanx.items():
        m = {i: np.mean(v) for i, v in d.items() if len(v) > 10}
        team_mean = np.mean(list(m.values())); deep_left = team_mean < Lm / 2
        gk[(p, away)] = min(m, key=m.get) if deep_left else max(m, key=m.get); right[(p, away)] = bool(deep_left)
    rules = ["all", "seen5", "seen7", "seen8", "seen9", "cover"]
    acc = {t: {r: {"length": [], "width": [], "line": []} for r in rules} for t in ("home", "away")}
    seen_n = []
    vi = 0
    for k in ks:
        p, ps, b = frames[k]
        if b is None: continue
        m, cx = views[vi % len(views)]; vi += 1
        inside = place(m, cx, b[0], Lv, Wv, Lm, Wm)
        for away in (False, True):
            t = "away" if away else "home"; key = (p, away)
            if key not in gk: continue
            pts = np.array([(x, y) for i, x, y in ps if (i >= 200) == away and i != gk[key]])
            if len(pts) < 8: continue
            ar = right[key] if attack_right is None else attack_right
            s_all = shape(pts, ar, Lm)
            for q, v in zip(("length", "width", "line"), s_all): acc[t]["all"][q].append(v)
            sel = inside(pts[:, 0], pts[:, 1]); sp = pts[sel]; seen_n.append(len(sp))
            if len(sp) >= 5:
                s = shape(sp, ar, Lm)
                for N in (5, 7, 8, 9):
                    if len(sp) >= N:
                        for q, v in zip(("length", "width", "line"), s): acc[t][f"seen{N}"][q].append(v)
                ok = covered(inside, sp, ar, Lm, Wm, margin)
                for q, v, o in zip(("length", "width", "line"), s, ok):
                    if o: acc[t]["cover"][q].append(v)
    out = {}
    for t, d in acc.items():
        out[t] = {}
        for r, qs in d.items():
            out[t][r] = {q: (round(float(np.median(v)), 1) if v else None) for q, v in qs.items()}
            out[t][r]["n"] = {q: len(v) for q, v in qs.items()}
    out["seen_per_team_median"] = float(np.median(seen_n)) if seen_n else None
    return out


def load_metrica(d, g):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import metricalab as ML
    home = ML.load_tracking(f"{d}/{g}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{d}/{g}/{g}_RawTrackingData_Away_Team.csv", 200)
    return {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home if k in away}


def dry():
    """stand-ins: a view that shows the left half only; a team spread 20-80 m -> seen shapes shorter than the truth"""
    rng = np.random.default_rng(1); fr = {}
    for k in range(600):
        ps = [(100 + i, 20 + 6 * i + rng.normal(0, 1), 5 + 6 * i) for i in range(11)] + [(200 + i, 25 + 6 * i, 4 + 6 * i) for i in range(11)]
        fr[k] = (1, ps, (52.5, 34.0))
    m = np.zeros((106, 33), bool); m[:53] = True
    return run(fr, [(m, 0.25)], 105.0, 64.0, step=1)


def main(a):
    if "--dry" in a: print(json.dumps(dry(), indent=1)); return
    out_p = a[a.index("--out") + 1] if "--out" in a else None; margin = float(a[a.index("--margin") + 1]) if "--margin" in a else 5.0
    pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--out", "--margin"))]
    d, clips = pos[0], pos[1:]
    res = {}
    for g in ("Sample_Game_1", "Sample_Game_2"):
        fr = load_metrica(d, g)
        for c in clips:
            v, Lv, Wv = view_masks(c)
            r = run(fr, v, Lv, Wv, margin=margin); res[f"{g[-1]}_{os.path.basename(os.path.dirname(c))}"] = r
            print(g, os.path.basename(os.path.dirname(c)), f"{len(v)} views, seen/team", r["seen_per_team_median"])
            for t in ("home", "away"):
                print("  ", t, " | ".join(f"{q}: " + " ".join(f"{rr}={r[t][rr][q]}({r[t][rr]['n'][q]})" for rr in ("all", "seen5", "seen8", "cover")) for q in ("length", "width", "line")))
    if out_p: json.dump(res, open(out_p, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])


# ---- 'hold' rule: a player who drops out of the picture keeps his last seen spot for hold_s seconds -------------
def detection_plan(n, fps, rng, runs=None, gaps=None):
    """per frame: which track id a player is seen under when he is in view (-1 = in a detection gap). With runs/gaps
    (seconds, drawn from our clips) every seen run is a NEW id, like our broken tracks; without, one id, never a gap."""
    tid = np.full(n, -1, int)
    if runs is None: tid[:] = 0; return tid
    i = int(rng.integers(0, int(fps))); c = 0
    while i < n:
        r = max(2, int(round(rng.choice(runs) * fps))); tid[i:i + r] = c; c += 1; i += r
        i += max(4, int(round(rng.choice(gaps) * fps)))
    return tid


def run_hold(frames, views, Lv, Wv, holds=(0, 2, 5, 10), Lm=105.0, Wm=68.0, seed=0, step=5, breaks=None, jit=None, every=2):
    """like run(), time-ordered: shape numbers from seen players + the last seen spot of players out of the picture
    (held for each hold_s); breaks = (runs, gaps) from our clips -> each seen run a new id (ghost of the old id kept).
    -> {team: {hold: {length, width, line, n, err: paired median abs error per frame}}} + truth"""
    rng = np.random.default_rng(seed); ks = sorted(frames)[::step]; fps = 25.0
    meanx = {}
    for k in ks[::4]:
        p, ps, _ = frames[k]
        for i, x, y in ps: meanx.setdefault((p, i >= 200), {}).setdefault(i, []).append(x)
    gk, right = {}, {}
    for (p, away), d in meanx.items():
        m = {i: np.mean(v) for i, v in d.items() if len(v) > 10}; deep_left = np.mean(list(m.values())) < Lm / 2
        gk[(p, away)] = min(m, key=m.get) if deep_left else max(m, key=m.get); right[(p, away)] = bool(deep_left)
    allids = sorted({i for k in ks for i, _, _ in frames[k][1]}); idx = {k: j for j, k in enumerate(ks)}
    plan = {i: detection_plan(len(ks), fps / step, rng, *(breaks or (None, None))) for i in allids}
    res = {t: {h: {"length": [], "width": [], "line": [], "err": {"length": [], "width": [], "line": []}} for h in ["truth", *holds]} for t in ("home", "away")}
    ghosts = {t: {h: {} for h in holds} for t in ("home", "away")}; vi = 0; prev_p = None
    for j, k in enumerate(ks):
        p, ps, b = frames[k]
        if p != prev_p: ghosts = {t: {h: {} for h in holds} for t in ("home", "away")}; prev_p = p
        if b is None: continue
        if j % every == 0: vi += 1
        m, cx = views[vi % len(views)]; inside = place(m, cx, b[0], Lv, Wv, Lm, Wm); now = j * step / fps
        for away in (False, True):
            t = "away" if away else "home"; key = (p, away)
            if key not in gk: continue
            team = [(i, x, y) for i, x, y in ps if (i >= 200) == away and i != gk[key]]
            if len(team) < 8: continue
            ar = right[key]; pts = np.array([(x, y) for _, x, y in team])
            truth = shape(pts, ar, Lm)
            for q, v in zip(("length", "width", "line"), truth): res[t]["truth"][q].append(v)
            vis = inside(pts[:, 0], pts[:, 1]); seen = {}
            for (i, x, y), v in zip(team, vis):
                c = plan[i][j]
                if v and c >= 0:
                    e = jit[rng.integers(0, len(jit))] if jit is not None else 0.0
                    seen[(i, c)] = np.array([x, y]) + e
            for h in holds:
                G = ghosts[t][h]
                for tid, pos in seen.items(): G[tid] = (pos, now)
                for tid in [q for q, (_, t0) in G.items() if now - t0 > h + 1e-9]: del G[tid]
                P = np.array([pos for tid, (pos, t0) in G.items()]) if h else (np.array(list(seen.values())) if seen else np.zeros((0, 2)))
                if len(P) >= 5:
                    s = shape(P, ar, Lm)
                    for q, v, tv in zip(("length", "width", "line"), s, truth): res[t][h][q].append(v); res[t][h]["err"][q].append(abs(v - tv))
    out = {}
    for t, d in res.items():
        out[t] = {}
        for h, qs in d.items():
            out[t][str(h)] = {q: (round(float(np.median(qs[q])), 1) if qs[q] else None) for q in ("length", "width", "line")}
            out[t][str(h)]["n"] = len(qs["length"])
            if h != "truth": out[t][str(h)]["err"] = {q: (round(float(np.median(v)), 1) if v else None) for q, v in qs["err"].items()}
    return out


def pipeline_check(frames, views, Lv, Wv, holds=(0, 5), Lm=105.0, Wm=68.0, seed=0, step=5, breaks=None, jit=None, every=2):
    """the SHIPPED code (analytics.held_points + the stats() rule) on Metrica seen through our views and track breaks:
    builds pipeline rows [track id, team, metres, px, box, keeper] per sampled frame and returns the team medians per
    hold next to the truth (all outfield players)"""
    from ipanema import analytics as AN
    rng = np.random.default_rng(seed); ks = sorted(frames)[::step]; fps = 25.0 / step
    meanx = {}
    for k in ks[::4]:
        p, ps, _ = frames[k]
        for i, x, y in ps: meanx.setdefault((p, i >= 200), {}).setdefault(i, []).append(x)
    gk, right = {}, {}
    for (p, away), d in meanx.items():
        m = {i: np.mean(v) for i, v in d.items() if len(v) > 10}; deep_left = np.mean(list(m.values())) < Lm / 2
        gk[(p, away)] = min(m, key=m.get) if deep_left else max(m, key=m.get); right[(p, away)] = bool(deep_left)
    allids = sorted({i for k in ks for i, _, _ in frames[k][1]})
    plan = {i: detection_plan(len(ks), fps, rng, *(breaks or (None, None))) for i in allids}
    per = {}; truth = {"A": {"length": [], "width": [], "line": []}, "B": {"length": [], "width": [], "line": []}}; vi = 0; prev_p = None; dirs = []
    for j, k in enumerate(ks):
        p, ps, b = frames[k]; rows = []
        if j % every == 0: vi += 1
        if p != prev_p: per[len(per)] = []; prev_p = p                     # a blank frame between halves
        if b is not None:
            m, cx = views[vi % len(views)]; inside = place(m, cx, b[0], Lv, Wv, Lm, Wm)
            for away, tm in ((False, "A"), (True, "B")):
                key = (p, away)
                if key not in gk: continue
                team = [(i, x, y) for i, x, y in ps if (i >= 200) == away]
                out = np.array([(x, y) for i, x, y in team if i != gk[key]])
                if len(out) >= 8:
                    for q, v in zip(("length", "width", "line"), shape(out, right[key], Lm)): truth[tm][q].append(v)
                for i, x, y in team:
                    c = plan[i][j]
                    if c >= 0 and inside(np.array([x]), np.array([y]))[0]:
                        e = jit[rng.integers(0, len(jit))] if jit is not None else 0.0
                        rows.append([i * 1000 + c, tm, np.array([x, y]) + e, None, None, i == gk[key]])
        dirs.append(right.get((p, False), True)); per[len(per)] = rows
    res = {"truth": {tm: {q: round(float(np.median(v)), 1) for q, v in d.items()} for tm, d in truth.items()}}
    for h in holds:
        held = AN.held_points(per, fps, h, L=Lm, W=Wm)
        r = {}
        for tm in ("A", "B"):
            lens, wids, hts = [], [], []
            for k in range(len(per)):
                pts = np.array([q[2] for q in per[k] if q[1] == tm and not q[5]])
                ar = dirs[min(k, len(dirs) - 1)] if tm == "A" else not dirs[min(k, len(dirs) - 1)]
                if len(pts) >= 5:
                    hts.append(pts[:, 0].min() if ar else Lm - pts[:, 0].max())
                    if not h: lens.append(np.ptp(pts[:, 0])); wids.append(np.ptp(pts[:, 1]))
                hp = held[k][tm]
                if h and hp is not None and len(hp) >= 5: lens.append(np.ptp(hp[:, 0])); wids.append(np.ptp(hp[:, 1]))
            r[tm] = {"length": round(float(np.median(lens)), 1), "width": round(float(np.median(wids)), 1), "line": round(float(np.median(hts)), 1), "n": len(lens)}
        res[str(h)] = r
    return res
