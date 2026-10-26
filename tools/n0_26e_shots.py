"""N0-26e: is our own shot detector right? stats.metrics.shots comes from Veo's highlight list when a match has one; without
it (any new upload, or a match whose Veo zip is missing) the app shows OUR detector (ipanema/metrics.shots: the ball
suddenly > 16 m/s, within 15 deg of the goal the team on the ball attacks, from within 30 m, and play stops or the ball
crosses the goal line within 3 s). On our matches it is only logged against Veo; never checked on pro data before.

Truth = Metrica's labelled SHOT events (team, start frame, start x/y, outcome: goal / saved / off target / blocked /
woodwork). Ours = metrics.shots on the pipeline's own inputs (tools/metricalab.ours: state, metres ball, ball speed,
restarts, attack direction), perfect ball + our 'streak' / 'flicker' ball noise (2 seeds), 4 halves.
Matching: a true shot <-> nearest unused one of ours, same team, within 2 s.

    PYTHONPATH=. python tools/n0_26e_shots.py <metrica sample-data/data dir> [--out results/qa/n0-26e/metrica_shots.json]
"""
import sys, os, csv, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P, metrics as M

L, W, FPS = 105.0, 68.0, 25.0
TOL = 2 * FPS


def outcome_of(sub):
    s = sub.upper()
    if "GOAL" in s and "ON TARGET" in s: return "goal"
    if "SAVED" in s: return "on target"           # ours: 'on target' = crossed the goal line; Metrica 'saved' = on target
    if "BLOCKED" in s: return "blocked"
    return "off target"                            # off target / woodwork / head-off


def truth_shots(path, per):
    out = []
    for r in csv.DictReader(open(path)):
        if int(r["Period"]) != per or r["Type"] != "SHOT" or r["Team"] not in ("Home", "Away"): continue
        x = float(r["Start X"]) * L if r["Start X"] not in ("", "NaN") else None; y = float(r["Start Y"]) * W if r["Start Y"] not in ("", "NaN") else None
        out.append({"frame": int(r["Start Frame"]), "team": "A" if r["Team"] == "Home" else "B", "x_m": x, "y_m": y,
                    "outcome": outcome_of(r["Subtype"]), "sub": r["Subtype"]})
    return out


def capture(fr, per, noisy, seed):
    """run metricalab.ours and keep what metrics.shots needs, exactly as ipanema/run.py feeds it"""
    import metricalab as ML
    got = {}; oc, ops, orst = P.carriers, P.pipeline_state, P.restarts
    def cc(*a, **k):
        r = oc(*a, **k); got["ballm"] = r[1]; return r
    def cps(*a, **k):
        r = ops(*a, **k); got["state"], got["bspeed"] = r[0], r[1]; return r
    def cr(*a, **k):
        r = orst(*a, **k); got["restarts"] = r; got["stops"] = P.stoppage_spans(a[0], a[1], a[2], a[3], a[4]); return r
    P.carriers, P.pipeline_state, P.restarts = cc, cps, cr
    try: o = ML.ours(fr, per, noisy, seed=seed)
    finally: P.carriers, P.pipeline_state, P.restarts = oc, ops, orst
    k0 = min(k for k in fr if fr[k][0] == per)
    return got, o["attack_right"], k0


def match(truth, ours, k0, tol=TOL):
    pairs = sorted((abs(int(round(o["t"] * FPS)) + k0 - t["frame"]), i, j) for i, t in enumerate(truth) for j, o in enumerate(ours)
                   if o["team"] == t["team"] and abs(int(round(o["t"] * FPS)) + k0 - t["frame"]) <= tol)
    ut, uo, out = set(), set(), []
    for d, i, j in pairs:
        if i in ut or j in uo: continue
        ut.add(i); uo.add(j); out.append((i, j))
    return out


def any_team_hit(truth, ours, k0, tol=TOL):
    """ours within 2 s of a real shot, any team (to tell 'wrong team' from 'not a shot')"""
    tf = [t["frame"] for t in truth]
    return sum(1 for o in ours if any(abs(int(round(o["t"] * FPS)) + k0 - f) <= tol for f in tf))


def run_one(fr, per, truth, noisy, seed, variants):
    got, ar, k0 = capture(fr, per, noisy, seed)
    row = {"noise": str(noisy), "seed": seed, "truth": len(truth), "attack_right": ar}
    for name, kw in variants.items():
        ours = shots_variant(got, ar, **kw)
        m = match(truth, ours, k0)
        loc = [float(np.hypot(ours[j]["x_m"] - truth[i]["x_m"], ours[j]["y_m"] - truth[i]["y_m"])) for i, j in m if truth[i]["x_m"] is not None]
        oc = [(truth[i]["outcome"], ours[j]["outcome"]) for i, j in m]
        row[name] = {"ours": len(ours), "found": len(m), "real": len(m), "near_any_team": any_team_hit(truth, ours, k0),
                     "loc_err_m_median": round(float(np.median(loc)), 1) if loc else None,
                     "outcome_right": sum(1 for a, b in oc if a == b or (a == "blocked" and b == "saved or blocked")), "outcome_pairs": oc,
                     "goals_truth": sum(1 for t in truth if t["outcome"] == "goal"), "goals_ours": sum(1 for o in ours if o["goal"]),
                     "goals_right": sum(1 for i, j in m if truth[i]["outcome"] == "goal" and ours[j]["goal"]),
                     "per_team": {tm: {"truth": sum(1 for t in truth if t["team"] == tm), "ours": sum(1 for o in ours if o["team"] == tm)} for tm in ("A", "B")},
                     "missed": [{"frame": truth[i]["frame"], "team": truth[i]["team"], "sub": truth[i]["sub"]} for i in range(len(truth)) if i not in {a for a, _ in m}],
                     "fakes_t": [o["t"] for j, o in enumerate(ours) if j not in {b for _, b in m}]}
    return row


def shots_variant(got, ar, **kw):
    """lab: metrics.shots with explicit rules (kw = end / goal_rule / aim / min_speed); stops always handed in like run.py"""
    return M.shots(got["state"], got["ballm"], got["bspeed"], FPS, L, W, ar, got["restarts"], stops=got["stops"], **kw)


VARIANTS = {"old": {"end": "state", "goal_rule": "unknown"},                      # before N0-26e
            "default": {"end": "state", "goal_rule": "centre"},                  # N0-26e default: goal from a centre-spot restart
            "stops": {"end": "stops", "goal_rule": "centre"},                    # option IPANEMA_SHOT_END=stops
            "stops_mouth": {"end": "stops", "goal_rule": "centre", "aim": "mouth"},
            "stops_look15": {"end": "stops", "goal_rule": "centre", "team_rule": "look1.5"},
            "stops_goalteam": {"end": "stops", "goal_rule": "centre", "team_rule": "goal"}}


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), noises=(False, "streak", "flicker"), seeds=(0, 1)):
    import metricalab as ML
    rows = []
    for g in games:
        gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
        fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}
        for per in halves:
            T = truth_shots(f"{gd}/{g}_RawEventsData.csv", per)
            for noisy in noises:
                for seed in (seeds if noisy else (0,)):
                    row = dict(game=g[-1], half=per, **run_one(fr, per, T, noisy, seed, VARIANTS))
                    rows.append(row); print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ("outcome_pairs", "missed", "fakes_t")}) for k, v in row.items()}), flush=True)
    return rows


def summary(rows):
    out = {}
    for v in VARIANTS:
        for nz in ("False", "streak", "flicker"):
            R = [r for r in rows if r["noise"] == nz]
            if not R: continue
            tr = sum(r["truth"] for r in R); ou = sum(r[v]["ours"] for r in R); fo = sum(r[v]["found"] for r in R)
            locs = [r[v]["loc_err_m_median"] for r in R if r[v]["loc_err_m_median"] is not None]
            out[f"{v}/{nz}"] = {"runs": len(R), "truth": tr, "ours": ou, "found": fo, "found_pct": round(100 * fo / max(1, tr)), "real_pct": round(100 * fo / max(1, ou)),
                                "near_real_shot_any_team": sum(r[v]["near_any_team"] for r in R), "count_ratio": round(ou / max(1, tr), 2),
                                "outcome_right": f"{sum(r[v]['outcome_right'] for r in R)}/{fo}", "goals": f"{sum(r[v]['goals_ours'] for r in R)} vs {sum(r[v]['goals_truth'] for r in R)}", "goals_right": sum(r[v]["goals_right"] for r in R),
                                "loc_err_m_median": round(float(np.median(locs)), 1) if locs else None}
    return out


if __name__ == "__main__":
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-26e/metrica_shots.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = main(a[0]); S = summary(rows); print(json.dumps(S, indent=1))
    json.dump({"rows": rows, "summary": S}, open(out, "w"), indent=1)
