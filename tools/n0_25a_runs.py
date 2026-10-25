"""N0-25a: are the app's runs (stats.metrics.runs: sprints >= 5.5 m/s for >= 0.8 s, the app's "Runs" layer) right?
Checked on Metrica pro data (every player at 25 fps = the truth).

Truth: each Metrica player's full track, smoothed 0.4 s (as n0_22c's truth), sprint = >= 5.5 m/s for >= 0.8 s.
Ours: the same player cut into seen runs and gaps drawn from OUR app clips, our measured jitter added, id swaps to a
team-mate after a gap (n0_22c.degrade), then measured by the rule under test:
  old  - metrics.runs as shipped up to 24 Oct: raw frame-to-frame speed (no smoothing), 5.5-10 m/s
  new  - metrics.runs as of N0-25a: speed from positions smoothed 0.8 s within each seen run (analytics.track_motion)
  + lab variants (smoothing window, min duration).
A truth sprint is found when one of ours on the same physical player overlaps it in time (1 s slack); one of ours is
real when it overlaps a truth sprint of the player whose positions it actually carries (after a swap: the mate).

    PYTHONPATH=. python tools/n0_25a_runs.py /tmp/metrica/data <picker_inputs.pkl>... [--game 1] [--minutes 20] [--out f.json]
    PYTHONPATH=. python tools/n0_25a_runs.py --dry
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import metrics as M
import n0_22c_metrica as C

AR = {"A": True, "B": False}


def truth_sprints(P, fps, speed=5.5, min_s=0.8):
    import pandas as pd
    Q = pd.DataFrame(P).rolling(max(1, int(0.4 * fps)), center=True, min_periods=1).mean().values
    s = np.r_[0, np.linalg.norm(np.diff(Q, axis=0), axis=1) * fps]
    return _segments(np.arange(len(P)), s >= speed, fps, min_s)


def _segments(ks, on, fps, min_s):
    out = []; a = None
    for i in range(len(ks)):
        if on[i] and (a is None or ks[i] - ks[i - 1] <= 3):
            if a is None: a = i
        else:
            if a is not None and (ks[i - 1] - ks[a]) / fps >= min_s: out.append((int(ks[a]), int(ks[i - 1])))
            a = i if on[i] else None
    if a is not None and (ks[-1] - ks[a]) / fps >= min_s: out.append((int(ks[a]), int(ks[-1])))
    return out


def degrade_src(P, fps, runs, gaps, jit, rng, mates, swap_p):
    """C.degrade + which physical player each kept row is (-1 = the player himself, i = mates[i])"""
    n = len(P); keep = np.zeros(n, bool); src = np.zeros(n, int); i = int(rng.integers(0, int(fps))); cur = -1
    while i < n:
        r = max(2, int(round(rng.choice(runs) * fps))); keep[i:i + r] = True; src[i:i + r] = cur; i += r
        i += max(4, int(round(rng.choice(gaps) * fps)))
        if mates and rng.random() < swap_p: cur = int(rng.integers(0, len(mates)))
    ks = np.flatnonzero(keep)
    Q = np.array([P[k] if src[k] < 0 else mates[src[k]][k] for k in ks]) + jit[rng.integers(0, len(jit), len(ks))]
    return ks, Q, src[ks]


RULES = {"old": dict(smooth=False), "new": dict(smooth=True), "smooth_only": dict(smooth=True, jump_m=0), "merge0.5": dict(smooth=True, merge_s=0.5), "jump3": dict(smooth=True, jump_m=3.0),
         "win0.4": dict(smooth=True, win_s=0.4), "win1.2": dict(smooth=True, win_s=1.2),
         "new_min1.0": dict(smooth=True, min_s=1.0), "new_min0.6": dict(smooth=True, min_s=0.6),
         "join0.5": dict(smooth=True, gap_join_s=0.5), "join1.0": dict(smooth=True, gap_join_s=1.0), "join0.5_min0.6": dict(smooth=True, gap_join_s=0.5, min_s=0.6)}


def ours(ks, Q, fps, rule):
    per = {int(k): [(1, "A", Q[i])] for i, k in enumerate(ks)}
    return [(int(round(r["t"] * fps)), int(round(r["t_end"] * fps)), r["distance_m"]) for r in M.runs(per, fps, np.full(int(ks[-1]) + 1, 2), AR, **RULES[rule])]


def overlap(a, b, slack): return a[0] <= b[1] + slack and b[0] <= a[1] + slack


def run(players, fps, runs, gaps, jit, seed=0, swap_p=0.25):
    rng = np.random.default_rng(seed); ids = list(players)
    T = {pid: truth_sprints(P, fps) for pid, P in players.items()}
    tot = {"truth": sum(len(v) for v in T.values()), "truth_dist_m": round(sum(float(np.linalg.norm(players[p][b] - players[p][a])) for p, v in T.items() for a, b in v))}
    res = {r: {"n": 0, "real": 0, "found": 0, "dist_m": 0.0} for r in RULES}; sl = int(fps); got = {r: {q: [] for q in ids} for r in RULES}
    for pid, P in players.items():
        mates_ids = [q for q in ids if q != pid and (q >= 100) == (pid >= 100)]; mates = [players[q] for q in mates_ids]
        ks, Q, src = degrade_src(P, fps, runs, gaps, jit, rng, mates, swap_p)
        for r in RULES:
            o = ours(ks, Q, fps, r); res[r]["n"] += len(o); res[r]["dist_m"] += sum(x[2] for x in o)
            for a, b, _ in o:
                who = src[np.searchsorted(ks, a)]; owner = pid if who < 0 else mates_ids[who]
                res[r]["real"] += any(overlap((a, b), t, sl) for t in T[owner]); got[r][owner].append((a, b))
    for r in RULES:   # found: a truth sprint overlapped by one of ours that carries that player (any track, after swaps too)
        res[r]["found"] = sum(any(overlap(t, o, sl) for o in got[r][q]) for q in ids for t in T[q])
    out = {"truth_sprints": tot["truth"], "truth_dist_m": tot["truth_dist_m"]}
    for r, v in res.items():
        out[r] = {"ours": v["n"], "ours_vs_truth_pct": round(100 * v["n"] / max(1, tot["truth"])), "found_pct": round(100 * v["found"] / max(1, tot["truth"])),
                  "real_pct": round(100 * v["real"] / v["n"]) if v["n"] else None, "dist_m": round(v["dist_m"])}
    return out


if __name__ == "__main__":
    a = sys.argv[1:]; out_path = None; minutes = 20.0; game = 1; swaps = [0.0, 0.25]
    if "--out" in a: i = a.index("--out"); out_path = a[i + 1]; del a[i:i + 2]
    if "--game" in a: i = a.index("--game"); game = int(a[i + 1]); del a[i:i + 2]
    if "--minutes" in a: i = a.index("--minutes"); minutes = float(a[i + 1]); del a[i:i + 2]
    if "--dry" in a:
        fps = 25.0; t = np.arange(int(120 * fps)) / fps
        players = {i: np.c_[50 + 30 * np.sin(t / (3 + i)), 34 + 10 * np.cos(t / (7 + i))] for i in range(4)}
        rng = np.random.default_rng(1); runs, gaps, jit = np.array([2.0, 4.0, 8.0]), np.array([0.2, 0.4]), rng.normal(0, 0.1, (500, 2))
    else:
        players, fps = C.metrica_players(a[0], minutes, game); runs, gaps, jit = C.our_breaks(a[1:])
        print(f"{len(players)} players x {minutes} min; our clips: {len(runs)} runs (median {np.median(runs):.2f} s), jitter sd {np.std(jit):.2f} m")
    res = {f"swap{sw}_seed{s}": run(players, fps, runs, gaps, jit, seed=s, swap_p=sw) for sw in swaps for s in range(2)}
    for s, r in res.items(): print(s, json.dumps(r))
    if out_path: json.dump({"game": game, "players": len(players), "minutes": minutes, "results": res}, open(out_path, "w"), indent=1)
