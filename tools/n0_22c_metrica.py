"""N0-22c: which way of measuring per-player distance / high-intensity distance in analytics.stats comes closest to the
truth when tracks are broken like ours? Metrica's free pro tracking (every player at 25 fps) is the truth; each player's
track is cut into seen runs and gaps drawn from OUR clips (run / gap lengths from the app clips' picker inputs, real
jitter added), then measured three ways:
  old  - one rolling mean over all rows (positions from after a gap leak into the frames before it), gaps not counted
  run  - rolling mean per seen run, gaps not counted
  line - per run + a gap of at most 2.5 s counted as a straight line when that speed is possible (cap 10 / 8 / 7 / 6 m/s;
         line_cap7 = the new app default)
Id swaps: with chance --swaps a track goes on with a team-mate after a gap (0.25 gives our clips' share of impossible gap jumps).
Usage: python tools/n0_22c_metrica.py <metrica data dir> <picker_inputs.pkl>... [--minutes 10] [--game 1|2] [--out f.json]
       python tools/n0_22c_metrica.py --dry   (stand-in tracks, for the test suite)"""
import sys, os, json, pickle, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import analytics as AN


def our_breaks(paths):
    """run lengths (frames) and gap lengths (s) and jitter (m) from real app clips"""
    import pandas as pd
    runs, gaps, jit = [], [], []
    for p in paths:
        d = pickle.load(open(p, "rb")); per, fps = d["per"], d["fps"]; tr = collections.defaultdict(dict)
        for k in range(len(per)):
            for r in per[k]: tr[r[0]][k] = np.asarray(r[2], float)
        for v in tr.values():
            ks = np.array(sorted(v)); P = np.array([v[k] for k in ks]); g = np.diff(ks); cut = np.flatnonzero(g > 3) + 1
            gaps += list(g[g > 3] / fps)
            for a, b in zip(np.r_[0, cut], np.r_[cut, len(ks)]):
                runs.append((b - a) / fps)
                if b - a >= 15:
                    Q = P[a:b]; sm = pd.DataFrame(Q).rolling(9, center=True, min_periods=9).mean().values; e = Q - sm
                    jit += [x for x in e[~np.isnan(e[:, 0])]]
    return np.array(runs), np.array(gaps), np.array(jit)


MODES = {"old": dict(runs=False, gap_line_s=0), "run": dict(runs=True, gap_line_s=0),
         "line": dict(runs=True, gap_line_s=2.5, gap_max_ms=10.0), "line_cap8": dict(runs=True, gap_line_s=2.5, gap_max_ms=8.0),
         "line_cap7": dict(runs=True, gap_line_s=2.5, gap_max_ms=7.0), "line_cap6": dict(runs=True, gap_line_s=2.5, gap_max_ms=6.0),
         "line_cap7_1s": dict(runs=True, gap_line_s=1.0, gap_max_ms=7.0)}   # line_cap7 = the new default


def measure(ks, P, fps, mode):
    """the app's per-player distance code (analytics.track_motion) in the mode under test"""
    kw = MODES[mode]
    _, step, hi = AN.track_motion(ks, P, fps, **kw)
    return float(step.sum()), float(hi.sum())


def truth(P, fps):
    import pandas as pd
    Q = pd.DataFrame(P).rolling(max(1, int(0.4 * fps)), center=True, min_periods=1).mean().values
    s = np.linalg.norm(np.diff(Q, axis=0), axis=1) * fps; s = np.where(s < 0.5, 0, np.minimum(s, 12))
    return float((s / fps).sum()), float((s[s > 5.5] / fps).sum())


def degrade(P, fps, runs, gaps, jit, rng, mates=None, swap_p=0.0):
    """keep seen runs / drop gaps with lengths drawn from ours; add our jitter. swap_p: chance that after a gap the
    track goes on with a team-mate (mates: list of (n, 2) arrays), like our stitched tracks / id swaps"""
    n = len(P); keep = np.zeros(n, bool); src = np.zeros(n, int); i = int(rng.integers(0, int(fps))); cur = -1
    while i < n:
        r = max(2, int(round(rng.choice(runs) * fps))); keep[i:i + r] = True; src[i:i + r] = cur; i += r
        i += max(4, int(round(rng.choice(gaps) * fps)))
        if mates and rng.random() < swap_p: cur = int(rng.integers(0, len(mates)))
    ks = np.flatnonzero(keep)
    Q = np.array([P[k] if src[k] < 0 else mates[src[k]][k] for k in ks]) + jit[rng.integers(0, len(jit), len(ks))]
    return ks, Q


def gap_speeds(ks, Q, fps):
    Ps = AN.smooth_runs(ks, Q, int(0.8 * fps), runs=True); out = []
    for i in np.flatnonzero(np.diff(ks) > 3) + 1:
        g = (ks[i] - ks[i - 1]) / fps
        if g <= 2.5: out.append(np.linalg.norm(Ps[i] - Ps[i - 1]) / g)
    return out


def run(players, fps, runs, gaps, jit, seed=0, swap_p=0.0):
    rng = np.random.default_rng(seed); tot = {"truth": [0, 0], **{m: [0, 0] for m in MODES}}; per_player = []; gsp = []
    for pid, P in players.items():
        mates = [Q for q, Q in players.items() if q != pid and (q >= 100) == (pid >= 100)]
        td, th = truth(P, fps); ks, Q = degrade(P, fps, runs, gaps, jit, rng, mates, swap_p); gsp += gap_speeds(ks, Q, fps)
        row = {"id": pid, "truth": [round(td), round(th)]}; tot["truth"][0] += td; tot["truth"][1] += th
        for m in MODES:
            d, h = measure(ks, Q, fps, m); tot[m][0] += d; tot[m][1] += h; row[m] = [round(d), round(h)]
        per_player.append(row)
    out = {m: {"distance_m": round(v[0]), "high_intensity_m": round(v[1]),
               "distance_vs_truth_pct": round(100 * v[0] / max(1e-9, tot["truth"][0])), "hi_vs_truth_pct": round(100 * v[1] / tot["truth"][1]) if tot["truth"][1] > 1 else None} for m, v in tot.items()}
    for m in MODES:   # per-player error (what a player row would show)
        e = [abs(r[m][0] - r["truth"][0]) / max(1, r["truth"][0]) for r in per_player]; out[m]["player_distance_err_median_pct"] = round(100 * float(np.median(e)))
    gsp = np.array(gsp); out["gaps"] = {"n": len(gsp), "speed_median": round(float(np.median(gsp)), 1), "over_7_pct": round(100 * float(np.mean(gsp > 7))), "over_10_pct": round(100 * float(np.mean(gsp > 10)))}
    return out


def metrica_players(data_dir, minutes, game=1):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import metricalab as ML
    g = os.path.join(data_dir, f"Sample_Game_{game}"); out = {}
    for side, team in (("Home", 0), ("Away", 100)):
        tr = ML.load_tracking(os.path.join(g, f"Sample_Game_{game}_RawTrackingData_{side}_Team.csv"), team)
        frs = sorted(f for f in tr if tr[f][0] == 1)[: int(minutes * 60 * ML.FPS)]
        seen = collections.defaultdict(list)
        for f in frs:
            for pid, x, y in tr[f][1]: seen[pid].append((f, x, y))
        for pid, rows in seen.items():
            if len(rows) == len(frs): out[pid] = np.array([[x, y] for _, x, y in rows])
    return out, ML.FPS


if __name__ == "__main__":
    a = sys.argv[1:]; out_path = None; minutes = 10.0
    if "--out" in a: i = a.index("--out"); out_path = a[i + 1]; del a[i:i + 2]
    game = 1; swaps = [0.0, 0.15, 0.3]
    if "--swaps" in a: i = a.index("--swaps"); swaps = [float(x) for x in a[i + 1].split(",")]; del a[i:i + 2]
    if "--game" in a: i = a.index("--game"); game = int(a[i + 1]); del a[i:i + 2]
    if "--minutes" in a: i = a.index("--minutes"); minutes = float(a[i + 1]); del a[i:i + 2]
    if "--dry" in a:
        fps = 25.0; t = np.arange(int(120 * fps)) / fps; rng = np.random.default_rng(1)
        players = {i: np.c_[50 + 20 * np.sin(t / (5 + i)), 34 + 10 * np.cos(t / (7 + i))] for i in range(4)}
        runs, gaps, jit = np.array([0.5, 1.0, 3.0]), np.array([0.2, 0.4, 1.5]), rng.normal(0, 0.1, (500, 2))
    else:
        players, fps = metrica_players(a[0], minutes, game); runs, gaps, jit = our_breaks(a[1:])
        print(f"breaks from our clips: {len(runs)} runs (median {np.median(runs):.2f} s), {len(gaps)} gaps (median {np.median(gaps):.2f} s), jitter rows {len(jit)}")
    res = {f"swap{sw}_seed{s}": run(players, fps, runs, gaps, jit, seed=s, swap_p=sw) for sw in swaps for s in range(2)}
    for s, r in res.items(): print(s, json.dumps(r))
    if out_path: json.dump({"game": game, "players": len(players), "minutes": minutes, "results": res}, open(out_path, "w"), indent=1)
