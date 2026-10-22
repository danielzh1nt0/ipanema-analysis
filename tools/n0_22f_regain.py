"""N0-22f: is the app's counter-press figure ('regained within 5 s', stats.json teams[].regained_within_5s_pct, and
'lost back in 5 s') right? Truth = Metrica's labelled events: after a BALL LOST by a team, does that team have its
next on-ball chain (tools/metricalab.truth_sequences) starting within 5 s? Ours = the pipeline's turnovers after
P.confirm_losses, run on Metrica's positions (clean and with our 'streak' noise) by tools/metricalab.ours.

    PYTHONPATH=. python tools/n0_22f_regain.py <metrica sample-data/data dir> [--out f.json]
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P


def truth_regains(ev, per, fps, win_s=5.0):
    """-> list of (loss frame, loser A/B, regained bool, winner chain length s, winner chain has pass)"""
    import metricalab as ML
    seqs = ML.truth_sequences(ev, per); E = [e for e in ev if e["period"] == per]
    passes = {"A": [e["f0"] for e in E if e["type"] == "PASS" and e["team"] == "Home"], "B": [e["f0"] for e in E if e["type"] == "PASS" and e["team"] == "Away"]}
    out = []
    for e in E:
        if e["type"] != "BALL LOST" or e["team"] not in ("Home", "Away"): continue
        lo = "A" if e["team"] == "Home" else "B"; wi = "B" if lo == "A" else "A"
        back = next((s for s in seqs if s["start"] > e["f0"] and s["team"] == lo), None)
        nxt = next((s for s in seqs if s["start"] >= e["f0"] - fps and s["team"] == wi), None)
        reg = back is not None and back["start"] - e["f0"] <= win_s * fps
        wl = (nxt["end"] - nxt["start"]) / fps if nxt else None
        wp = bool(nxt and any(nxt["start"] <= f <= nxt["end"] for f in passes[wi]))
        out.append({"f": e["f0"], "lost_by": lo, "regained": reg, "winner_chain_s": wl, "winner_pass": wp,
                    "back_s": round((back["start"] - e["f0"]) / fps, 2) if back else None})
    return out


def ours_tvs(fr, per, noisy, **kw):
    """the pipeline's kept turnovers (after confirm_losses) for one Metrica half"""
    import metricalab as ML
    got = {}
    orig = P.confirm_losses
    def cap(*a, **k):
        got["all"] = [dict(t) for t in a[0]]; r = orig(*a, **k); got["kept"], got["dropped"] = r; got["cstate"] = a[1]; return r
    P.confirm_losses = cap
    try: ML.ours(fr, per, noisy, **kw)
    finally: P.confirm_losses = orig
    return got


def regain_from_loss(t, cstate, fps, win_s=5.0):
    """loser back on the ball (counted state) within win_s of the moment it lost it (frame_lost), not of the win"""
    i = 0 if t["lost_by"] == "A" else 1; k = int(t.get("frame_lost", t["frame"]))
    return any(cstate[q] == i for q in range(k + 1, min(len(cstate), k + int(win_s * fps) + 1)))


def matched(T, o, fps, tol_s=2.0):
    """truth losses matched to our (all) turnovers of the same loser within tol_s -> agreement of the regain flag"""
    used = set(); agree = n = 0; rows = []
    for t in T:
        best = None
        for j, u in enumerate(o["all"]):
            if j in used or u["lost_by"] != t["lost_by"]: continue
            dt = abs(u["frame_lost"] - (t["f"] - o["f0"]))
            if dt <= tol_s * fps and (best is None or dt < best[0]): best = (dt, j)
        if best is None: continue
        used.add(best[1]); u = o["all"][best[1]]
        rows.append((t["regained"], regain_from_loss(u, o["cstate"], fps), bool(u["regained_within_5s"]), (u["frame"], u["lost_by"]) in {(k["frame"], k["lost_by"]) for k in o["kept"]}))
    return {"n": len(rows), "truth_regained": sum(r[0] for r in rows),
            "from_loss_agree": sum(r[0] == r[1] for r in rows), "old_agree": sum(r[0] == r[2] for r in rows),
            "from_loss_hits": sum(r[0] and r[1] for r in rows), "old_hits": sum(r[0] and r[2] for r in rows),
            "kept_n": sum(r[3] for r in rows), "kept_truth_regained": sum(r[0] for r in rows if r[3])}


def rate(xs):
    xs = list(xs); return round(100 * sum(xs) / len(xs), 1) if xs else None


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), noises=(False, "streak")):
    import metricalab as ML
    rows = []
    for g in games:
        gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
        fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}; ev = ML.load_events(f"{gd}/{g}_RawEventsData.csv")
        for per in halves:
            T = truth_regains(ev, per, ML.FPS)
            ts9 = [t for t in T if t["winner_chain_s"] is not None and (t["winner_chain_s"] >= 3.0 or t["winner_pass"])]
            for noisy in noises:
                o = ours_tvs(fr, per, noisy); o["f0"] = min(k for k in fr if fr[k][0] == per)
                row = {"game": g[-1], "half": per, "noise": str(noisy),
                       "truth_losses": len(T), "truth_regain5_pct": rate(t["regained"] for t in T),
                       "truth_s9_losses": len(ts9), "truth_s9_regain5_pct": rate(t["regained"] for t in ts9),
                       "ours_losses": len(o["kept"]), "ours_regain5_pct": rate(t["regained_within_5s"] for t in o["kept"]),
                       "ours_all_losses": len(o["all"]), "ours_all_regain5_pct": rate(t["regained_within_5s"] for t in o["all"]),
                       "ours_all_from_loss_pct": rate(regain_from_loss(t, o["cstate"], ML.FPS) for t in o["all"]),
                       "ours_kept_from_loss_pct": rate(regain_from_loss(t, o["cstate"], ML.FPS) for t in o["kept"]),
                       "matched": matched(T, o, ML.FPS),
                       "truth_back_s_hist": np.histogram([t["back_s"] for t in T if t["back_s"] is not None and t["back_s"] <= 10], bins=[0, 1, 2, 3, 4, 5, 10])[0].tolist()}
                rows.append(row); print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-22f/metrica_regain.json"
    rows = main(a[0]); json.dump(rows, open(out, "w"), indent=1)
