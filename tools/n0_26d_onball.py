"""N0-26d: are the app's per-player 'time on the ball' and 'passes received' right? stats.json players[].time_on_ball_s
(every frame the player is the 'carrier' = nearest to the ball within 2.5 m, possession.carriers) and
players[].passes_received (completed passes from passes() whose 'to' is the player). Never checked on pro data before.

Truth (Metrica's labelled events):
  passes received = PASS events with To = player (Metrica's PASS = completed; a failed one is BALL LOST).
  time on ball    = per spell, from the moment the ball reaches the player (end frame of a PASS to him, or the start of
                    his RECOVERY / SET PIECE / CARRY) to the moment he plays it (start frame of his PASS / SHOT / BALL
                    LOST); one-touch play = 0 s; spells capped at 20 s (a keeper holding before a restart).
Ours = the pipeline on Metrica's positions (tools/metricalab.ours), perfect ball + our 'streak' / 'flicker' ball noise
(2 seeds), 4 halves = 20 runs. Variants for time on ball:
  old    = today's rule (every carrier frame)
  state  = carrier frames only while the spell state gives his team the ball (as passes / touches use it)
  eps    = carrier frames inside the cleaned contacts (analytics.touch_episodes with the spell state, = N0-26c touches)
Scores per team-half (players with >= 5 real touches): ours / truth total, per-player median |error| %, Spearman,
share error (each player's share of the team's figure, median |error| in points).

    PYTHONPATH=. python tools/n0_26d_onball.py <metrica sample-data/data dir> [--out results/qa/n0-26d/metrica_onball.json]
"""
import sys, os, json, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import analytics as AN, possession as P

FPS = 25.0
END = {"PASS", "SHOT", "BALL LOST"}
CAP_S = 20.0


def _pid(team, s):
    return (100 if team == "Home" else 200) + int(s[6:])


def truth(path, per):
    """-> (time on ball s per player, passes received per player, real touches per player)"""
    E = [r for r in csv.DictReader(open(path)) if int(r["Period"]) == per and r["Team"] in ("Home", "Away")]
    E.sort(key=lambda r: (int(r["Start Frame"]), int(r["End Frame"] or 0)))
    tob, rec, tch = {}, {}, {}; holder, since = None, None
    for r in E:
        t, f0, f1 = r["Type"], int(r["Start Frame"]), int(r["End Frame"] or 0)
        p = _pid(r["Team"], r["From"]) if r["From"].startswith("Player") else None
        if p is None: continue
        if t in ("RECOVERY", "SET PIECE", "CARRY") and holder != p: holder, since = p, f0
        if t in END:
            tch[p] = tch.get(p, 0) + 1
            if holder == p and since is not None: tob[p] = tob.get(p, 0.0) + min(CAP_S, max(0.0, (f0 - since) / FPS))
            else: tob.setdefault(p, 0.0)
            holder, since = None, None
            if t == "PASS" and r["To"].startswith("Player"):
                q = _pid(r["Team"], r["To"]); rec[q] = rec.get(q, 0) + 1; holder, since = q, f1
    return tob, rec, tch


def capture(fr, per, noisy, seed):
    import metricalab as ML
    got = {}; oc, osp, opa = P.carriers, P.spell_state, AN.passes
    def cc(*a, **k):
        r = oc(*a, **k); got["frames_"] = r[0]; return r
    def cs(*a, **k):
        r = osp(*a, **k); got["cstate"] = r; return r
    def cp(*a, **k):
        r = opa(*a, **k); got["passes"] = r[0]; return r
    P.carriers, P.spell_state, AN.passes = cc, cs, cp
    try: ML.ours(fr, per, noisy, seed=seed)
    finally: P.carriers, P.spell_state, AN.passes = oc, osp, opa
    return got["frames_"], got["cstate"], got["passes"]


def onball_old(frames_, fps=FPS):
    out = {}
    for f in frames_:
        if f["carrier"] is not None: out[int(f["carrier"])] = out.get(int(f["carrier"]), 0) + 1 / fps
    return out


def onball_state(frames_, state, fps=FPS):
    out = {}
    for k, f in enumerate(frames_):
        c = f["carrier"]
        if c is None or k >= len(state) or int(state[k]) not in (0, 1) or f["team"] != ("A" if int(state[k]) == 0 else "B"): continue
        out[int(c)] = out.get(int(c), 0) + 1 / fps
    return out


def received(passes_):
    out = {}
    for p in passes_:
        if p.get("completed") and p.get("to") is not None: out[int(p["to"])] = out.get(int(p["to"]), 0) + 1
    return out


def spearman(a, b):
    from scipy.stats import spearmanr
    return round(float(spearmanr(a, b)[0]), 2) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else None


def score(T, O, team, ids):
    ids = [i for i in ids if (i < 200) == (team == "A")]
    if len(ids) < 3: return None
    t = np.array([T.get(i, 0) for i in ids], float); o = np.array([O.get(i, 0) for i in ids], float)
    tot_t = sum(v for k, v in T.items() if (k < 200) == (team == "A")); tot_o = sum(v for k, v in O.items() if (k < 200) == (team == "A"))
    m = t > 0
    return {"players": len(ids), "truth_total": round(float(tot_t), 1), "ours_total": round(float(tot_o), 1), "ratio": round(tot_o / max(1e-6, tot_t), 2),
            "median_abs_err_pct": round(float(np.median(np.abs(o[m] - t[m]) / t[m]) * 100)) if m.any() else None, "spearman": spearman(t, o),
            "share_err_pts": round(float(np.median(np.abs(o / max(1e-6, o.sum()) - t / max(1e-6, t.sum())))) * 100, 1)}


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), noises=(False, "streak", "flicker"), seeds=(0, 1)):
    import metricalab as ML
    rows = []
    for g in games:
        gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
        fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}
        for per in halves:
            Tt, Tr, Tc = truth(f"{gd}/{g}_RawEventsData.csv", per); ids = sorted(p for p, v in Tc.items() if v >= 5)
            for noisy in noises:
                for seed in (seeds if noisy else (0,)):
                    frames_, cstate, ps = capture(fr, per, noisy, seed)
                    E = AN.touch_episodes(frames_, FPS, state=cstate)
                    V = {"old": onball_old(frames_), "state": onball_state(frames_, cstate), "eps": AN.touch_time(frames_, E, FPS)}
                    row = {"game": g[-1], "half": per, "noise": str(noisy), "seed": seed,
                           "time_on_ball": {v: {tm: score(Tt, O, tm, ids) for tm in ("A", "B")} for v, O in V.items()},
                           "passes_received": {tm: score(Tr, received(ps), tm, ids) for tm in ("A", "B")}}
                    rows.append(row); print(json.dumps(row), flush=True)
    return rows


def summary(rows):
    out = {}
    for key, vs in (("time_on_ball", ("old", "state", "eps")), ("passes_received", (None,))):
        for v in vs:
            for nz in ("False", "streak", "flicker"):
                S = [s for r in rows if r["noise"] == nz for s in ((r[key][v] if v else r[key]).values()) if s]
                if not S: continue
                out[f"{key}/{v or '-'}/{nz}"] = {"ratio": [min(s["ratio"] for s in S), round(float(np.median([s["ratio"] for s in S])), 2), max(s["ratio"] for s in S)],
                                                  "median_abs_err_pct": round(float(np.median([s["median_abs_err_pct"] for s in S if s["median_abs_err_pct"] is not None]))),
                                                  "spearman": round(float(np.median([s["spearman"] for s in S if s["spearman"] is not None])), 2),
                                                  "share_err_pts": round(float(np.median([s["share_err_pts"] for s in S])), 1)}
    return out


if __name__ == "__main__":
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-26d/metrica_onball.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = main(a[0]); S = summary(rows); print(json.dumps(S, indent=1))
    json.dump({"rows": rows, "summary": S}, open(out, "w"), indent=1)
