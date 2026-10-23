"""N0-23a: is the app's transition figure right? stats.json teams[].forward_within_3s_pct ("forward pass within 3 s of
winning the ball") and time_to_forward_pass_median_s. Truth = Metrica's labelled events: after a BALL LOST by the other
team, the winner's first completed PASS that gains >= 5 m towards the opponent goal (end x - start x), timed at its End
Frame (the app times it when the receiver has the ball), from the winner's RECOVERY. Ours = the pipeline's turnovers after
P.confirm_losses (what stats() gets), run on Metrica's positions with a perfect ball and with our 'streak' / 'flicker'
ball noise (tools/metricalab.ours).

    PYTHONPATH=. python tools/n0_23a_transition.py /tmp/metrica/data [--out results/qa/n0-23a/metrica_transition.json]
"""
import sys, os, csv, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P

L, FPS = 105.0, 25.0
ONBALL = ("PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST", "CHALLENGE")


def load_events_xy(path):
    ev = []
    for r in csv.DictReader(open(path)):
        f = lambda s: None if s in ("", "NaN") else float(s)
        ev.append({"team": r["Team"], "type": r["Type"], "sub": r["Subtype"], "period": int(r["Period"]), "f0": int(r["Start Frame"]),
                   "f1": int(r["End Frame"] or 0), "x0": f(r["Start X"]), "x1": f(r["End X"])})
    return ev


def home_attacks_right(frames, per):
    """Home's mean x at the period's first frame < half way -> Home attacks +x"""
    k = min(k for k in frames if frames[k][0] == per); ps = frames[k][1]
    xs = [x for pid, x, y in ps if pid < 200]; return float(np.mean(xs)) < L / 2


def truth_transitions(ev, per, home_right, fwd_m=5.0, win_s=3.0, look_s=8.0, recov_s=5.0):
    """-> one row per BALL LOST: winner, win frame, time of the winner's first forward pass (None = none within look_s
    or before the winner's chain ends)."""
    E = sorted([e for e in ev if e["period"] == per], key=lambda e: (e["f0"], e["f1"])); out = []
    for i, e in enumerate(E):
        if e["type"] != "BALL LOST" or e["team"] not in ("Home", "Away"): continue
        wt = "Away" if e["team"] == "Home" else "Home"; sign = 1 if (wt == "Home") == home_right else -1
        later = [q for q in E[i + 1:] if q["f0"] >= e["f0"]]
        win = next((q for q in later if q["team"] == wt and q["type"] == "RECOVERY" and q["f0"] - e["f0"] <= recov_s * FPS), None)
        if win is None: win = next((q for q in later if q["team"] == wt and q["type"] in ONBALL), None)
        if win is None: continue
        t_fwd = None; first_pass = None
        for q in later:
            if q["f0"] < win["f0"]: continue
            if q["type"] == "BALL OUT": break
            if q["team"] not in ("Home", "Away") or q["type"] not in ONBALL: continue
            if q["team"] != wt and q["type"] in ("RECOVERY", "PASS", "CARRY", "SHOT", "SET PIECE"): break   # other team has it
            if q["f0"] - win["f0"] > look_s * FPS: break
            if q["team"] == wt and q["type"] == "PASS" and q["x0"] is not None and q["x1"] is not None:
                gain = (q["x1"] - q["x0"]) * L * sign
                if first_pass is None: first_pass = q
                if gain >= fwd_m:
                    t_fwd = round((max(q["f1"], q["f0"]) - win["f0"]) / FPS, 2); break
        out.append({"f_lost": e["f0"], "f_won": win["f0"], "won_by": "A" if wt == "Home" else "B", "sub": e["sub"],
                    "time_to_forward_pass": t_fwd if t_fwd is not None and t_fwd <= look_s else None})
    return out


def ours_kept(fr, per, noisy, seed=0):
    import metricalab as ML
    got = {}; orig = P.confirm_losses
    def cap(*a, **k):
        r = orig(*a, **k); got["kept"] = r[0]; got["all"] = [dict(t) for t in a[0]]; return r
    P.confirm_losses = cap
    try: ML.ours(fr, per, noisy, seed=seed)
    finally: P.confirm_losses = orig
    return got


def fw3(rows):
    v = [(t["time_to_forward_pass"] is not None and t["time_to_forward_pass"] <= 3) for t in rows]
    return round(100 * float(np.mean(v)), 1) if v else None


def med(rows):
    v = [t["time_to_forward_pass"] for t in rows if t["time_to_forward_pass"] is not None]
    return round(float(np.median(v)), 2) if v else None


def matched(T, kept, f0, tol_s=2.0):
    """truth wins matched to our kept turnovers (same winner, win frames within tol_s) -> yes/no agreement"""
    used = set(); rows = []
    for t in T:
        best = None
        for j, u in enumerate(kept):
            if j in used or u["won_by"] != t["won_by"]: continue
            dt = abs(f0 + u["frame"] - t["f_won"])
            if dt <= tol_s * FPS and (best is None or dt < best[0]): best = (dt, j)
        if best is None: continue
        used.add(best[1]); u = kept[best[1]]
        a = t["time_to_forward_pass"] is not None and t["time_to_forward_pass"] <= 3
        b = u["time_to_forward_pass"] is not None and u["time_to_forward_pass"] <= 3
        rows.append((a, b))
    return {"n": len(rows), "truth_yes": sum(a for a, b in rows), "ours_yes": sum(b for a, b in rows),
            "both_yes": sum(a and b for a, b in rows), "agree": sum(a == b for a, b in rows)}


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), noises=(False, "streak", "flicker"), seeds=(0, 1)):
    import metricalab as ML
    rows = []
    for g in games:
        gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
        fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}; ev = load_events_xy(f"{gd}/{g}_RawEventsData.csv")
        for per in halves:
            T = truth_transitions(ev, per, home_attacks_right(fr, per)); f0 = min(k for k in fr if fr[k][0] == per)
            for noisy in noises:
                for seed in (seeds if noisy else (0,)):
                    o = ours_kept(fr, per, noisy, seed)
                    row = {"game": g[-1], "half": per, "noise": str(noisy), "seed": seed,
                           "truth_wins": len(T), "truth_fwd3_pct": fw3(T), "truth_median_s": med(T),
                           "ours_wins": len(o["kept"]), "ours_fwd3_pct": fw3(o["kept"]), "ours_median_s": med(o["kept"]),
                           "ours_all_wins": len(o["all"]), "ours_all_fwd3_pct": fw3(o["all"]),
                           "matched": matched(T, o["kept"], f0)}
                    rows.append(row); print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-23a/metrica_transition.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = main(a[0]); json.dump(rows, open(out, "w"), indent=1)
