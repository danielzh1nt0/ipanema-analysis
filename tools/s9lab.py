"""S9 (8 Oct): a duel is a lost ball only when the winner keeps it >= 3 s or plays a pass (P.confirm_losses).

    PYTHONPATH=. python tools/s9lab.py <metrica sample-data/data dir>   -> results/possession/s9/s9lab.json

1) Metrica (4 halves, clean + streak noise): balls lost before / after vs the labelled BALL LOST events, and vs the
   same S9 definition applied to the labels (the other team's next labelled chain lasts >= 3 s or has a pass).
2) Our three demo exports (results/volume/runs/matches/*): the exported state at ~10 fps -> spell state -> the rule on
   the exported turnovers + passes: losses per team per half before / after.
3) The 36 by-eye spell strips (results/kaggle/spell_zoom, SFK-BP): which turnovers at each spell's start/end the rule keeps.
"""
import sys, os, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P

OUT = "results/possession/s9"
MATCHES = ["SFKBP1109", "p15u-vs-aik-2026-09-21-bd09", "p15u-vs-vallentuna-2026-10-03-6cce"]

def match_pairs(a, b, tol):
    """a, b: [(frame, team)] -> number of a matched to a distinct b of the same team within tol"""
    pairs = sorted((abs(x - y), i, j) for i, (x, s) in enumerate(a) for j, (y, u) in enumerate(b) if s == u and abs(x - y) <= tol)
    ua, ub = set(), set()
    for _, i, j in pairs:
        if i in ua or j in ub: continue
        ua.add(i); ub.add(j)
    return len(ua)

def truth_losses(ev, per, fps, keep_s=3.0):
    """labelled BALL LOST events of one half -> (all, s9) as [(frame, 'A'/'B')]; s9 = the winner's next chain lasts
    >= keep_s or contains a pass"""
    import metricalab as ML
    seqs = ML.truth_sequences(ev, per); E = [e for e in ev if e["period"] == per]
    passes = {"A": [e["f0"] for e in E if e["type"] == "PASS" and e["team"] == "Home"], "B": [e["f0"] for e in E if e["type"] == "PASS" and e["team"] == "Away"]}
    allx, s9 = [], []
    for e in E:
        if e["type"] != "BALL LOST" or e["team"] not in ("Home", "Away"): continue
        tm = "A" if e["team"] == "Home" else "B"; wi = "B" if tm == "A" else "A"; allx.append((e["f0"], tm))
        nxt = next((s for s in seqs if s["start"] >= e["f0"] - fps and s["team"] == wi), None)
        if nxt is None: continue
        if (nxt["end"] - nxt["start"]) >= keep_s * fps or any(nxt["start"] <= f <= nxt["end"] for f in passes[wi]): s9.append((e["f0"], tm))
    return allx, s9

def metrica(d):
    import metricalab as ML
    rows = []
    for g in ("Sample_Game_1", "Sample_Game_2"):
        gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
        fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}; ev = ML.load_events(f"{gd}/{g}_RawEventsData.csv")
        for per in (1, 2):
            tall, ts9 = truth_losses(ev, per, ML.FPS)
            for noisy in (False, "streak", "flicker"):
                for keep in (0.0, 2.0, 3.0, 4.0):
                    O = ML.ours(fr, per, noisy, loss_keep_s=keep); lf = O["loss_frames"]; tol = 2 * ML.FPS
                    rows.append({"game": g[-1], "half": per, "noise": str(noisy), "keep_s": keep, "ours": len(lf), "truth": len(tall), "truth_s9": len(ts9),
                                 "real": match_pairs(lf, tall, tol), "found": match_pairs(tall, lf, tol), "found_s9": match_pairs(ts9, lf, tol), "real_s9": match_pairs(lf, ts9, tol)})
                    print(json.dumps(rows[-1]), flush=True)
    return rows

def load_export(m):
    base = f"results/volume/runs/matches/{m}"; md = json.load(open(f"{base}/match_data.json")); st = json.load(open(f"{base}/stats.json"))
    t, s = [], []
    for f in sorted(glob.glob(f"{base}/frames_*.json")):
        for x in json.load(open(f))["frames"]:
            t.append(x["t"]); s.append({"A": 0, "B": 1}.get(x.get("possession"), 2))
    return md, st, np.array(t), np.array(s)

def export_run(m, keep_s=P.LOSS_KEEP_S):
    """the rule on one exported match, only inside the time covered by the local frame chunks"""
    md, st, t, s = load_export(m)
    if not len(t): return None
    fps = 1.0 / float(np.median(np.diff(t))); cstate = P.spell_state(s, fps)
    lo, hi = t[0] + 10, t[-1] - 10; tvs = []
    for tv in md["turnovers"]:
        if not (lo <= tv["t_won"] <= hi): continue
        k = int(np.searchsorted(t, tv["t_won"])); tvs.append(dict(tv, frame=k, frame_lost=int(np.searchsorted(t, tv["t"])), orig_frame=tv["frame"]))
    # sanity: the rebuilt spell state should give the winner at the exported win time
    agree = float(np.mean([cstate[min(len(cstate) - 1, tv["frame"])] == {"A": 0, "B": 1}[tv["won_by"]] for tv in tvs])) if tvs else None
    kept, dropped = P.confirm_losses([dict(x) for x in tvs], cstate, st["passes"], fps, keep_s)
    per = lambda L, tm: sum(1 for x in L if x["lost_by"] == tm)
    mins = sum(max(0.0, min(hi, p["t_end"]) - max(lo, p["t_start"])) for p in md["periods"]) / 60
    return {"match": m, "minutes_covered": round(mins, 1), "fps": round(fps, 2), "state_agrees_at_win": round(agree, 2) if agree is not None else None,
            "before": {"A": per(tvs, "A"), "B": per(tvs, "B")}, "after": {"A": per(kept, "A"), "B": per(kept, "B")},
            "confirmed_by": {k: sum(1 for x in kept if x.get("confirmed") == k) for k in ("hold", "pass", "clip end")},
            "dropped_back": sum(1 for x in dropped if x.get("dropped", "").startswith("duel, ball straight")),
            "kept": kept, "dropped": dropped}

def spells(res):
    sp = json.load(open("results/review/spells/moments.json"))["spells"]; eye = json.load(open(f"{OUT}/eye_grades.json"))
    allt = sorted(res["kept"] + res["dropped"], key=lambda x: x["t_won"]); out = []
    for s in sp:
        start = [x for x in allt if x["won_by"] == s["team"] and abs(x["t_won"] - s["t_start"]) <= 2.0] if s["before"] != s["team"] else []
        end = [x for x in allt if x["lost_by"] == s["team"] and abs(x["t"] - s["t_end"]) <= 2.0] if s["after"] != s["team"] else []
        f = lambda L: ("kept" if "confirmed" in L[0] else "dropped") if L else "none"
        out.append({"id": s["id"], "team": s["team"], "dur": s["dur"], "change": f"{s['before']}>{s['team']}>{s['after']}", "eye": eye.get(s["id"], "?"),
                    "turnover_in": f(start), "turnover_out": f(end)})
    return out

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); rep = {}
    ex = {m: export_run(m) for m in MATCHES}
    rep["exports"] = {m: {k: v for k, v in r.items() if k not in ("kept", "dropped")} for m, r in ex.items() if r}
    for m, r in rep["exports"].items(): print(m, json.dumps(r))
    if os.path.exists(f"{OUT}/eye_grades.json"):
        rep["spells"] = spells(ex["SFKBP1109"])
        for r in rep["spells"]: print(r)
    if len(sys.argv) > 1: rep["metrica"] = metrica(sys.argv[1])
    json.dump(rep, open(f"{OUT}/s9lab.json", "w"), indent=1)
    json.dump({m: {"kept": r["kept"], "dropped": r["dropped"]} for m, r in ex.items() if r}, open(f"{OUT}/export_turnovers.json", "w"), indent=0, default=str)
