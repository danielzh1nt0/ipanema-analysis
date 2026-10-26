"""N0-26c: are the app's per-player touches right? stats.json players[].touches (the app's player table reads it) =
number of runs of frames where the player is the 'carrier' (nearest to the ball within 2.5 m, possession.carriers),
a new run after a gap > 0.4 s. Never checked on pro data before.

Truth (Metrica's labelled events): one spell on the ball ends with exactly one PASS, SHOT or BALL LOST by that player,
so truth touches per player = count of those events (From = player) in the half.
Ours = the pipeline's carriers on Metrica's positions (tools/metricalab.ours), perfect ball and our 'streak' /
'flicker' ball noise (2 seeds each). Variants:
  old    = today's rule (analytics.stats)
  eps    = the cleaned contacts passes() links into passes (analytics.touch_episodes, run.py's spell state)
  eps0   = same, no state filter
  eps_<s> = eps, only contacts lasting >= s seconds (analytics.touch_counts; 0.4 = the new default)
Scores per team-half: total ours / truth, per-player median |error| %, Spearman rank correlation, top player right.

    PYTHONPATH=. python tools/n0_26c_touches.py <metrica sample-data/data dir> [--out results/qa/n0-26c/metrica_touches.json]
"""
import sys, os, json, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import analytics as AN, possession as P

FPS = 25.0
END = {"PASS", "SHOT", "BALL LOST"}


def truth_touches(path, per):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["Team"] not in ("Home", "Away") or int(r["Period"]) != per or r["Type"] not in END or not r["From"].startswith("Player"): continue
        pid = (100 if r["Team"] == "Home" else 200) + int(r["From"][6:]); out[pid] = out.get(pid, 0) + 1
    return out


def old_touches(frames_, fps=FPS):
    """today's analytics.stats rule, frame index = frame number (every carrier frame is a visible frame)"""
    on = {}
    for k, f in enumerate(frames_):
        if f["carrier"] is not None: on.setdefault(f["carrier"], []).append(k)
    return {t: sum(1 for i in range(1, len(v)) if v[i] - v[i - 1] > int(0.4 * fps)) + 1 for t, v in on.items()}


def eps_touches(eps):
    out = {}
    for e in eps: out[int(e[2])] = out.get(int(e[2]), 0) + 1
    return out


def capture(fr, per, noisy, seed):
    import metricalab as ML
    got = {}; oc, osp = P.carriers, P.spell_state
    def cc(*a, **k):
        r = oc(*a, **k); got["frames_"] = r[0]; return r
    def cs(*a, **k):
        r = osp(*a, **k); got["cstate"] = r; return r
    P.carriers, P.spell_state = cc, cs
    try: ML.ours(fr, per, noisy, seed=seed)
    finally: P.carriers, P.spell_state = oc, osp
    return got["frames_"], got["cstate"]


def spearman(a, b):
    from scipy.stats import spearmanr
    return round(float(spearmanr(a, b)[0]), 2) if len(a) > 2 else None


def score(T, O, team):
    ids = sorted(t for t in T if (t < 200) == (team == "A") and T[t] >= 5)        # players with >= 5 real spells
    if not ids: return None
    t = np.array([T[i] for i in ids], float); o = np.array([O.get(i, 0) for i in ids], float)
    tot_t = sum(v for k, v in T.items() if (k < 200) == (team == "A")); tot_o = sum(v for k, v in O.items() if (k < 200) == (team == "A"))
    return {"players": len(ids), "truth_total": int(tot_t), "ours_total": int(tot_o), "ratio": round(tot_o / max(1, tot_t), 2),
            "median_abs_err_pct": round(float(np.median(np.abs(o - t) / t) * 100)), "spearman": spearman(t, o),
            "top_right": ids[int(np.argmax(t))] == ids[int(np.argmax(o))], "share_err_pts": round(float(np.median(np.abs(o / max(1, o.sum()) - t / t.sum()))) * 100, 1)}


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), noises=(False, "streak", "flicker"), seeds=(0, 1)):
    import metricalab as ML
    rows = []
    for g in games:
        gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
        fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}
        for per in halves:
            T = truth_touches(f"{gd}/{g}_RawEventsData.csv", per)
            for noisy in noises:
                for seed in (seeds if noisy else (0,)):
                    frames_, cstate = capture(fr, per, noisy, seed)
                    E = AN.touch_episodes(frames_, FPS, state=cstate)
                    V = {"old": old_touches(frames_), "eps": AN.touch_counts(E, FPS, 0), "eps0": AN.touch_counts(AN.touch_episodes(frames_, FPS), FPS, 0),
                         **{f"eps_{m}": AN.touch_counts(E, FPS, m) for m in (0.2, 0.3, 0.4, 0.5)}}
                    row = {"game": g[-1], "half": per, "noise": str(noisy), "seed": seed, "v": {v: {tm: score(T, O, tm) for tm in ("A", "B")} for v, O in V.items()}}
                    rows.append(row); print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-26c/metrica_touches.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = main(a[0]); json.dump(rows, open(out, "w"), indent=1)
