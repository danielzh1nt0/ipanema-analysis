"""N0-29a: are the app's PER-PLAYER pressure figures right? Never checked on pro data before.

What the app exports (stats.json players[], analytics.stats as ipanema/run.py calls it; both hidden in the app until M2):
  pressures_applied     - per opponent possession sequence (spell state), the player counts once if he comes within
                          2 m of the carrier (S7). N0-24d checked the TEAM sum (0.76-1.07x the truth), not the players.
  time_under_pressure_s - his seconds on the ball (carrier frames inside his cleaned contacts, N0-26d) with an
                          opponent within 2 m.

Metrica's 2 pro games, 4 halves, real ids (what M2 will give). Truth:
  same rule  - our definitions with perfect inputs: Metrica's sequences, the carrier from the events, 2 m from tracking
               (n0_24d_ppda.truth_carrier / the same per-sequence count, per player).
  def. acts  - the usual per-player defensive count from Metrica's events: tackles + ground challenges (won or lost) +
               interceptions + steals (RECOVERY THEFT) by that player. Tells whether 'pressures' ranks the players a
               coach would call the busy defenders.
  baseline   - 'who is near the play': seconds within 10 m of the true ball. If ours ranks players no better than this,
               the per-player figure says nothing beyond 'he was near the ball'.
Ours = the pipeline (metricalab.ours noise models: perfect, streak, flicker x 2 seeds) + analytics.stats with the touch
episodes like run.py.

Scores per team-half (outfield + keepers, players with >= 10 min): ratio = sum ours / sum truth; err = median per-player
|ours - truth| / truth (players with truth >= 5 pressures / >= 5 s); rank = Spearman vs the truth; top3 = how many of the
truth's 3 busiest are in ours' top 3.

    PYTHONPATH=. python tools/n0_29a_playerpress.py <metrica sample-data/data dir> [--out results/qa/n0-29a/metrica_playerpress.json]
    PYTHONPATH=. python tools/n0_29a_playerpress.py <dir> --refresh-ball 2.5   (truth carrier only with the ball within
        2.5 m, like the app's carrier; re-scores the saved runs, no pipeline re-run)
    PYTHONPATH=. python tools/n0_29a_playerpress.py --dry
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import analytics as AN
from n0_28e_passnet import spearman

L, W, FPS = 105.0, 68.0, 25.0
PRESS_R = 2.0
DEF_ACTS = {("CHALLENGE", "TACKLE"), ("CHALLENGE", "GROUND"), ("RECOVERY", "INTERCEPTION"), ("RECOVERY", "THEFT")}
MIN_PLAY_MIN = 10.0


def is_def_act(typ, sub):
    sub = sub or ""
    return any(typ == t and sub.startswith(s) for t, s in DEF_ACTS)


def truth_players(ev, fr, per, car, seqs, ball_m=None):
    """-> {pid: {press, under_s, onball_s, def_acts, near_s, play_s}} with the same rule as the app on perfect inputs.
    ball_m: the event carrier counts only while the tracked ball is within ball_m of him (the app's carrier needs the ball
    within 2.5 m; the event carrier alone also holds it through stoppages and while the ball travels)."""
    from n0_24d_ppda import pid_of
    T = {}
    def row(p): return T.setdefault(p, {"press": 0, "under_s": 0.0, "onball_s": 0.0, "def_acts": 0, "near_s": 0.0, "play_s": 0.0})
    ks = sorted(k for k in fr if fr[k][0] == per)
    for k in ks:
        _, ps, b = fr[k]
        for pid, x, y in ps:
            r = row(pid); r["play_s"] += 1 / FPS
            if b is not None and np.hypot(x - b[0], y - b[1]) <= 10: r["near_s"] += 1 / FPS
        c = car.get(k)
        if c is None: continue
        cp = next(((x, y) for pid, x, y in ps if pid == c), None)
        if cp is None or (ball_m is not None and (b is None or np.hypot(cp[0] - b[0], cp[1] - b[1]) > ball_m)): continue
        row(c)["onball_s"] += 1 / FPS
        od = [np.hypot(x - cp[0], y - cp[1]) for pid, x, y in ps if (pid >= 200) != (c >= 200)]
        if od and min(od) <= PRESS_R: row(c)["under_s"] += 1 / FPS
    for sq in seqs:
        seen = set()
        for k in range(sq["start"], sq["end"] + 1):
            c = car.get(k); f = fr.get(k)
            if c is None or f is None or (c < 200) != (sq["team"] == "A"): continue
            cp = next(((x, y) for pid, x, y in f[1] if pid == c), None)
            if cp is None or (ball_m is not None and (f[2] is None or np.hypot(cp[0] - f[2][0], cp[1] - f[2][1]) > ball_m)): continue
            for pid, x, y in f[1]:
                if (pid >= 200) != (c >= 200) and pid not in seen and np.hypot(x - cp[0], y - cp[1]) <= PRESS_R:
                    seen.add(pid); row(pid)["press"] += 1
    for r in ev:
        if int(r["Period"]) != per or r["Team"] not in ("Home", "Away"): continue
        if is_def_act(r["Type"], r["Subtype"]):
            p = pid_of(r["From"], r["Team"])
            if p is not None: row(p)["def_acts"] += 1
    return T


def team_scores(ours, truth, key_o, key_t, min_truth, ids):
    """ours / truth: {pid: value}; ids = the players of one team-half"""
    o = [float(ours.get(i, 0)) for i in ids]; t = [float(truth.get(i, 0)) for i in ids]
    errs = [abs(a - b) / b for a, b in zip(o, t) if b >= min_truth]
    top_t = set(sorted(ids, key=lambda i: -truth.get(i, 0))[:3]); top_o = set(sorted(ids, key=lambda i: -ours.get(i, 0))[:3])
    return {"ratio": round(sum(o) / sum(t), 2) if sum(t) else None, "err": round(float(np.median(errs)), 2) if errs else None,
            "rank": None if spearman(o, t) is None else round(spearman(o, t), 2), "top3": len(top_t & top_o), "n": len(ids)}


def share_scores(o, t):
    """under-pressure share of a player's time on the ball (players with >= 5 s on the ball in both): median abs error
    (pts), team mean ours / truth (pts)"""
    ids = [i for i in t if t[i] is not None and o.get(i) is not None]
    if not ids: return {"err_pts": None, "ours_pct": None, "truth_pct": None, "n": 0}
    return {"err_pts": round(float(np.median([abs(o[i] - t[i]) for i in ids])) * 100, 1), "ours_pct": round(float(np.mean([o[i] for i in ids])) * 100, 1),
            "truth_pct": round(float(np.mean([t[i] for i in ids])) * 100, 1), "n": len(ids)}


def score_half(st_players, T):
    """-> {team: {press_vs_same, press_vs_defacts, baseline_vs_defacts, under_vs_same, under_rank_vs_onball_truth}}"""
    O = {p["id"]: p for p in st_players}
    out = {}
    for tm, lo in (("A", lambda i: i < 200), ("B", lambda i: i >= 200)):
        ids = [i for i, r in T.items() if lo(i) and r["play_s"] >= MIN_PLAY_MIN * 60]
        op = {i: O[i]["pressures_applied"] if i in O else 0 for i in ids}
        ou = {i: O[i]["time_under_pressure_s"] if i in O else 0.0 for i in ids}
        ouo = {i: O[i].get("under_old_s", 0.0) if i in O else 0.0 for i in ids}
        def share(u, b): return {i: (u[i] / b[i] if b.get(i, 0) >= 5 else None) for i in ids}
        osh = share(ou, {i: O[i].get("time_on_ball_s", 0.0) if i in O else 0.0 for i in ids}); osho = share(ouo, {i: O[i].get("onball_old_s", 0.0) if i in O else 0.0 for i in ids})
        tsh = share({i: T[i]["under_s"] for i in ids}, {i: T[i]["onball_s"] for i in ids})
        tp = {i: T[i]["press"] for i in ids}; tu = {i: T[i]["under_s"] for i in ids}; td = {i: T[i]["def_acts"] for i in ids}; tn = {i: T[i]["near_s"] for i in ids}
        out[tm] = {"press_vs_same": team_scores(op, tp, "o", "t", 5, ids), "press_vs_defacts": team_scores(op, td, "o", "t", 3, ids),
                   "same_rule_vs_defacts": team_scores(tp, td, "o", "t", 3, ids), "near_ball_vs_defacts": team_scores(tn, td, "o", "t", 3, ids),
                   "near_ball_vs_same": team_scores(tn, tp, "o", "t", 5, ids),
                   "under_vs_same": team_scores(ou, tu, "o", "t", 5, ids), "near_ball_vs_under": team_scores(tn, tu, "o", "t", 5, ids),
                   "under_old_vs_same": team_scores(ouo, tu, "o", "t", 5, ids),
                   "share_vs_same": share_scores(osh, tsh), "share_old_vs_same": share_scores(osho, tsh),
                   "truth_press_total": sum(tp.values()), "truth_under_s_total": round(sum(tu.values()), 1), "truth_def_acts_total": sum(td.values())}
    return out


def ours_players(fr, per, noisy, seed):
    """analytics.stats like run.py (touch episodes on the spell state) on metricalab.ours' inputs -> players[]"""
    from n0_24d_ppda import ours_run
    st, got = ours_run(fr, per, noisy, seed)
    eps = AN.touch_episodes(got["frames_"], FPS, state=got["cstate"])
    st = AN.stats(got["per"], got["frames_"], got["tvs"], got["ps"], got["tracks"], got["state"], FPS, L, W, got["ar"], PRESS_R, 5.0, sequences_=got["seqs"], episodes_=eps)
    old = AN.stats(got["per"], got["frames_"], got["tvs"], got["ps"], got["tracks"], got["state"], FPS, L, W, got["ar"], PRESS_R, 5.0, sequences_=got["seqs"])   # lab: every carrier frame (N0-26d's old rule)
    O = {p["id"]: p for p in old["players"]}
    for p in st["players"]:
        o = O.get(p["id"], {}); p["under_old_s"] = o.get("time_under_pressure_s", 0.0); p["onball_old_s"] = o.get("time_on_ball_s", 0.0)
    return st["players"]


def summarise(rows):
    """mean of the team-half scores per noise"""
    out = {}
    for nz in sorted({r["noise"] for r in rows}):
        rs = [t for r in rows if r["noise"] == nz for t in r["teams"].values()]
        out[nz] = {}
        for k in ("press_vs_same", "press_vs_defacts", "same_rule_vs_defacts", "near_ball_vs_defacts", "near_ball_vs_same", "under_vs_same", "under_old_vs_same", "near_ball_vs_under", "share_vs_same", "share_old_vs_same"):
            d = {}
            for q in (("err_pts", "ours_pct", "truth_pct") if k.startswith("share") else ("ratio", "err", "rank", "top3")):
                v = [t[k][q] for t in rs if t[k][q] is not None]
                d[q] = round(float(np.mean(v)), 2) if v else None
            if not k.startswith("share"): d["rank_min"] = min((t[k]["rank"] for t in rs if t[k]["rank"] is not None), default=None)
            out[nz][k] = d
        out[nz]["team_halves"] = len(rs)
    return out


def dry():
    """stand-ins: 3 attackers (A) keep the ball, 3 defenders (B) chase at 1 / 3 / 8 m -> only the closest presses"""
    T = {101: {"press": 0, "under_s": 30.0, "onball_s": 60.0, "def_acts": 0, "near_s": 100.0, "play_s": 2700.0},
         201: {"press": 12, "under_s": 0.0, "onball_s": 0.0, "def_acts": 6, "near_s": 90.0, "play_s": 2700.0},
         202: {"press": 4, "under_s": 0.0, "onball_s": 0.0, "def_acts": 3, "near_s": 80.0, "play_s": 2700.0},
         203: {"press": 0, "under_s": 0.0, "onball_s": 0.0, "def_acts": 0, "near_s": 20.0, "play_s": 2700.0},
         204: {"press": 7, "under_s": 0.0, "onball_s": 0.0, "def_acts": 2, "near_s": 85.0, "play_s": 2700.0}}
    st = [{"id": 201, "pressures_applied": 10, "time_under_pressure_s": 0.0, "time_on_ball_s": 0.0}, {"id": 202, "pressures_applied": 5, "time_under_pressure_s": 0.0},
          {"id": 204, "pressures_applied": 6, "time_under_pressure_s": 0.0}, {"id": 101, "pressures_applied": 0, "time_under_pressure_s": 24.0, "time_on_ball_s": 48.0, "under_old_s": 30.0, "onball_old_s": 70.0}]
    return score_half(st, T)


def refresh(d, out, ball_m):
    """recompute the truth with ball_m (no pipeline re-run): ours from the saved per-player rows"""
    import n0_24d_ppda as PP, metricalab as ML
    J = json.load(open(out)); T_ = {}
    for g in sorted({r["game"] for r in J["runs"]}):
        fr, ev = PP.load_game(d, f"Sample_Game_{g}")
        for per in (1, 2):
            ks = sorted(k for k in fr if fr[k][0] == per); car = PP.truth_carrier(ev, per, ks[0], ks[-1])
            T_[(g, per)] = truth_players(ev, fr, per, car, ML.truth_sequences(PP.ML_events(ev), per), ball_m=ball_m)
    for r in J["runs"]:
        T = T_[(r["game"], r["half"])]
        pl = [{"id": int(i), "pressures_applied": p["ours_press"], "time_under_pressure_s": p["ours_under_s"], "time_on_ball_s": p["ours_onball_s"],
               "under_old_s": p["ours_under_old_s"], "onball_old_s": p["ours_onball_old_s"]} for i, p in r["players"].items()]
        r["teams"] = score_half(pl, T)
        for i, p in r["players"].items():
            t = T[int(i)]; p.update({"truth_press": t["press"], "truth_under_s": round(t["under_s"], 1), "truth_onball_s": round(t["onball_s"], 1)})
    J["summary"] = summarise(J["runs"]); J["truth_ball_m"] = ball_m
    print(json.dumps(J["summary"], indent=1)); json.dump(J, open(out, "w"), indent=1)


def main(d, out):
    import n0_24d_ppda as PP, metricalab as ML
    rows = []
    for g in ("Sample_Game_1", "Sample_Game_2"):
        fr, ev = PP.load_game(d, g)
        for per in (1, 2):
            ks = sorted(k for k in fr if fr[k][0] == per); car = PP.truth_carrier(ev, per, ks[0], ks[-1])
            seqs = ML.truth_sequences(PP.ML_events(ev), per)
            T = truth_players(ev, fr, per, car, seqs)
            for noisy in (False, "streak", "flicker"):
                for seed in ((0, 1) if noisy else (0,)):
                    pl = ours_players(fr, per, noisy, seed)
                    sc = score_half(pl, T)
                    row = {"game": g[-1], "half": per, "noise": str(noisy), "seed": seed, "teams": sc,
                           "players": {str(i): {"truth_press": T[i]["press"], "truth_under_s": round(T[i]["under_s"], 1), "def_acts": T[i]["def_acts"], "near_s": round(T[i]["near_s"], 1),
                                                "ours_press": next((p["pressures_applied"] for p in pl if p["id"] == i), 0),
                                                "ours_under_s": next((p["time_under_pressure_s"] for p in pl if p["id"] == i), 0.0),
                                                "ours_under_old_s": next((p["under_old_s"] for p in pl if p["id"] == i), 0.0),
                                                "truth_onball_s": round(T[i]["onball_s"], 1), "ours_onball_s": next((p["time_on_ball_s"] for p in pl if p["id"] == i), 0.0),
                                                "ours_onball_old_s": next((p["onball_old_s"] for p in pl if p["id"] == i), 0.0)} for i in T if T[i]["play_s"] >= MIN_PLAY_MIN * 60}}
                    rows.append(row)
                    print(g, per, noisy, seed, json.dumps({tm: {k: v for k, v in s.items() if k in ("press_vs_same", "press_vs_defacts", "under_vs_same")} for tm, s in sc.items()}), flush=True)
    summ = summarise(rows); print(json.dumps(summ, indent=1))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"summary": summ, "runs": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--dry" in a: print(json.dumps(dry(), indent=1)); sys.exit()
    out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-29a/metrica_playerpress.json"
    if "--refresh-ball" in a: refresh(a[0], out, float(a[a.index("--refresh-ball") + 1]))
    else: main(a[0], out)
