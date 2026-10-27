"""N0-27a part 2: does the 'a better pass was open' rule mean anything on real football? With PERFECT inputs (Metrica: all
22 players + the ball) the rule (analytics.best_option) is applied to every completed PASS and to every failed pass
(BALL LOST ... INTERCEPTION: intended receiver = the passer's team-mate nearest the spot where it was cut out).
A rule that says something should call a FAILED pass second best more often than a completed one. Variants of the rule
(margin, how far a better option may be, a lane that widens with the pass length) are compared the same way.

    PYTHONPATH=. python tools/n0_27a_rule.py /tmp/metrica/src/data [--out results/qa/n0-27a/metrica_rule.json]
"""
import sys, os, json, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import analytics as AN
from n0_27a_better_option import rows_of, L, W

VARIANTS = {"app": {}, "margin12": {"margin": 12.0}, "reach30": {"reach": 30.0}, "widen": {"widen": 0.1},
            "widen+reach30": {"widen": 0.1, "reach": 30.0}, "widen+reach30+margin12": {"widen": 0.1, "reach": 30.0, "margin": 12.0}}


def load(path):
    out = []
    for r in csv.DictReader(open(path)):
        if r["Team"] not in ("Home", "Away"): continue
        ok = r["Type"] == "PASS"; bad = r["Type"] == "BALL LOST" and "INTERCEPTION" in r["Subtype"]
        if not (ok or bad): continue
        f = lambda s: None if s in ("", "NaN") else float(s)
        out.append({"team": "A" if r["Team"] == "Home" else "B", "period": int(r["Period"]), "f0": int(r["Start Frame"]), "f1": int(r["End Frame"] or 0),
                    "x1": f(r["End X"]), "y1": f(r["End Y"]), "failed": bad})
    return out


def lanes_var(rows, me, team, sgn, widen=0.0, reach=45.0, lane_half=1.5, fwd_m=3.0):
    mr = next(r for r in rows if r[0] == me); c = mr[2]; opps = [r[2] for r in rows if r[1] != team]; out = []
    for r in rows:
        if r[1] != team or r[0] == me: continue
        d = float(np.linalg.norm(r[2] - c))
        if d > 45.0: continue
        out.append({"to": int(r[0]), "open": not AN._blocked(c, r[2], opps, lane_half + widen * d), "forward": bool((r[2][0] - c[0]) * sgn > fwd_m), "d": d})
    return out


def flag(rows, me, team, sgn, to, margin=AN.BETTER_MARGIN, reach=45.0, widen=0.0):
    ln = lanes_var(rows, me, team, sgn, widen=widen)
    opts = AN.option_values(rows, me, team, sgn, ln)
    if to not in opts: return None
    cand = {t: v for t, v in opts.items() if t == to or next(l for l in ln if l["to"] == t)["d"] <= reach}
    bt = max(cand, key=lambda t: cand[t]["value"])
    return bool(bt != to and cand[bt]["value"] >= cand[to]["value"] + margin)


def main(d, games=("Sample_Game_1", "Sample_Game_2")):
    import metricalab as ML
    from n0_23a_transition import home_attacks_right
    res = {v: {"completed": [0, 0], "failed": [0, 0]} for v in VARIANTS}; per_half = []
    for g in games:
        gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
        fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}; E = load(f"{gd}/{g}_RawEventsData.csv")
        for per in (1, 2):
            hr = home_attacks_right(fr, per); hh = {v: {"completed": [0, 0], "failed": [0, 0]} for v in VARIANTS}
            for p in E:
                if p["period"] != per or p["f0"] not in fr or fr[p["f0"]][2] is None or p["x1"] is None: continue
                _, ps, b = fr[p["f0"]]; A = p["team"] == "A"
                mine = [(np.hypot(x - b[0], y - b[1]), pid) for pid, x, y in ps if (pid < 200) == A]
                if not mine or min(mine)[0] > 2.5: continue
                me = min(mine)[1]; e = np.array([p["x1"] * L, p["y1"] * W]); fe = p["f1"] if p["f1"] in fr else p["f0"]
                rec = [(np.hypot(x - e[0], y - e[1]), pid) for pid, x, y in fr[fe][1] if (pid < 200) == A and pid != me]
                if not rec: continue
                to = min(rec)[1]; rows = rows_of(ps); sgn = 1 if A == hr else -1; k = "failed" if p["failed"] else "completed"
                for v, kw in VARIANTS.items():
                    f = flag(rows, me, p["team"], sgn, to, **kw)
                    if f is None: continue
                    for R in (res, hh): R[v][k][0] += 1; R[v][k][1] += int(f)
            per_half.append({"game": g[-1], "half": per, **{v: {k: round(100 * x[1] / max(1, x[0]), 1) for k, x in hh[v].items()} for v in VARIANTS}})
            print(json.dumps(per_half[-1]), flush=True)
    tot = {v: {k: {"n": x[0], "flagged": x[1], "pct": round(100 * x[1] / max(1, x[0]), 1)} for k, x in r.items()} for v, r in res.items()}
    return {"total": tot, "halves": per_half}


if __name__ == "__main__":
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-27a/metrica_rule.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    r = main(a[0]); json.dump(r, open(out, "w"), indent=1); print(json.dumps(r["total"]))
