"""N0-29c: per-player top speed and high-intensity distance (stats.json players[].top_speed_ms, high_intensity_m).
N0-22c fixed distance but left these two too high (top speeds 8+ m/s on most tracks = feet jitter; high-intensity
103-163% of the truth on Metrica). Which way of measuring them comes closest when tracks are broken like ours?

Truth: Metrica pro tracking (25 fps), positions smoothed 0.4 s; top speed = the best 1-s average speed (the usual GPS
definition, a single frame is never a top speed); high-intensity = distance at > 5.5 m/s (as N0-22c).
Ours: each player's track cut into seen runs / gaps drawn from our app clips, our feet jitter added, id swaps after a
gap with chance --swaps (0.25 = our clips' share of impossible gap jumps; tools/n0_22c_metrica.py).

Top speed modes:
  old    - the fastest single row speed (analytics.track_motion, 0.8 s smoothing)            [app before N0-29c]
  p95    - 95th percentile of row speeds
  sus1   - best 1-s average of row speeds, only over 1 s fully seen (no gap) inside one seen run; a raw > 2 m jump in
           <= 3 frames (another person, as metrics.runs) splits the run
  sus1_nojump / sus05 / sus2 - variants
High-intensity modes:
  old    - track_motion (seen steps > 5.5 m/s + gap lines > 5.5 m/s)
  sus1   - only seen stretches staying > 5.5 m/s for >= 1 s, plus gap lines whose seen neighbours are both > 5.5
  sus05  - the same with >= 0.5 s
  seen   - seen steps only, no gap line ever high-intensity
Scores per mode: total vs truth %, median per-player error, rank (Spearman) of players vs the truth.

Usage: python tools/n0_29c_speed.py <metrica data dir> <picker_inputs.pkl>... [--minutes 20] [--out f.json]
       python tools/n0_29c_speed.py --dry   (stand-in tracks, for the test suite)"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import analytics as AN
import n0_22c_metrica as B


def truth(P, fps):
    import pandas as pd
    Q = pd.DataFrame(P).rolling(max(1, int(0.4 * fps)), center=True, min_periods=1).mean().values
    s = np.linalg.norm(np.diff(Q, axis=0), axis=1) * fps; s = np.where(s < 0.5, 0, np.minimum(s, 12))
    w = int(fps); top = float(np.convolve(s, np.ones(w) / w, mode="valid").max()) if len(s) >= w else 0.0
    return top, float((s[s > 5.5] / fps).sum())


TOP = {"old": dict(old=True), "p95": dict(pct=95), "sus1": dict(win_s=1.0), "sus05": dict(win_s=0.5), "sus2": dict(win_s=2.0),
       "sus1_nojump": dict(win_s=1.0, jump_m=0), "sus1_j3": dict(win_s=1.0, jump_m=3.0), "sus05_nojump": dict(win_s=0.5, jump_m=0)}
HI = {"old": dict(old=True), "sus1": dict(min_s=1.0), "sus05": dict(min_s=0.5), "seen": dict(min_s=0.0, gap_hi=False)}


def measure(ks, P, fps):
    out = {"top": {}, "hi": {}}
    for m, kw in TOP.items(): out["top"][m] = AN.top_speed(ks, P, fps, **kw)
    for m, kw in HI.items(): out["hi"][m] = AN.high_intensity(ks, P, fps, **kw)
    return out


def spearman(a, b):
    from scipy.stats import spearmanr
    return round(float(spearmanr(a, b).correlation), 2) if len(a) > 2 else None


def run(players, fps, runs, gaps, jit, seed=0, swap_p=0.0):
    rng = np.random.default_rng(seed); rows = []
    for pid, P in players.items():
        mates = [Q for q, Q in players.items() if q != pid and (q >= 100) == (pid >= 100)]
        tt, th = truth(P, fps); ks, Q = B.degrade(P, fps, runs, gaps, jit, rng, mates, swap_p)
        rows.append({"id": pid, "truth": [round(tt, 2), round(th, 1)], **measure(ks, Q, fps)})
    T = np.array([r["truth"][0] for r in rows]); H = np.array([r["truth"][1] for r in rows]); out = {"top": {}, "hi": {}}
    for m in TOP:
        v = np.array([r["top"][m] for r in rows])
        out["top"][m] = {"median_ms": round(float(np.median(v)), 2), "truth_median_ms": round(float(np.median(T)), 2),
                         "err_median_ms": round(float(np.median(np.abs(v - T))), 2), "over_8_pct": round(100 * float(np.mean(v > 8))), "rank": spearman(v, T)}
    for m in HI:
        v = np.array([r["hi"][m] for r in rows])
        out["hi"][m] = {"vs_truth_pct": round(100 * float(v.sum() / max(1e-9, H.sum()))), "err_median_pct": round(100 * float(np.median(np.abs(v - H) / np.maximum(H, 10)))), "rank": spearman(v, H)}
    return out, rows


if __name__ == "__main__":
    a = sys.argv[1:]; out_path = None; minutes = 20.0; swaps = [0.0, 0.25]
    if "--out" in a: i = a.index("--out"); out_path = a[i + 1]; del a[i:i + 2]
    if "--minutes" in a: i = a.index("--minutes"); minutes = float(a[i + 1]); del a[i:i + 2]
    if "--dry" in a:
        fps = 25.0; t = np.arange(int(120 * fps)) / fps; rng = np.random.default_rng(1)
        players = {i: np.c_[50 + 20 * np.sin(t / (2 + i)), 34 + 10 * np.cos(t / (3 + i))] for i in range(4)}
        runs, gaps, jit = np.array([0.5, 1.5, 3.0]), np.array([0.2, 0.4, 1.5]), rng.normal(0, 0.1, (500, 2))
        print(json.dumps(run(players, fps, runs, gaps, jit)[0])); sys.exit()
    runs, gaps, jit = B.our_breaks(a[1:])
    print(f"breaks: {len(runs)} runs (median {np.median(runs):.2f} s), {len(gaps)} gaps, jitter sd {jit.std(0).round(2)}", flush=True)
    res = {}
    for game in (1, 2):
        players, fps = B.metrica_players(a[0], minutes, game)
        for sw in swaps:
            for s in range(2):
                r, rows = run(players, fps, runs, gaps, jit, seed=s, swap_p=sw); res[f"game{game}_swap{sw}_seed{s}"] = {"summary": r, "players": rows}
                print(game, sw, s, json.dumps(r), flush=True)
    if out_path: json.dump({"minutes": minutes, "results": res}, open(out_path, "w"), indent=1)
