"""N0-23c: are the app's pass-direction figures right? stats.json teams[].forward_pass_share_pct (share of a team's passes
that gain > 3 m towards the opponent goal), teams[].progressive_passes (completed passes gaining >= 10 m) and the same
per player. Never checked before.

Truth = Metrica's labelled PASS events (all completed; Start/End X of the BALL), gain = (End X - Start X) * 105 m towards
the team's attacking goal. Ours = analytics.passes as the pipeline runs it (tools/metricalab.ours: possession state,
turnovers, carriers) on Metrica's positions with a perfect ball and with our 'streak' / 'flicker' ball noise.
Per pass: ours matched to truth (same team, passer's last touch within 1 s of the PASS start) -> kind agreement, gain error.

    PYTHONPATH=. python tools/n0_23c_progressive.py /tmp/metrica/data [--out results/qa/n0-23c/metrica_progressive.json]
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import analytics as AN
from n0_23a_transition import load_events_xy, home_attacks_right
import csv

L, W, FPS = 105.0, 68.0, 25.0


def load_passes(path):
    out = []
    for r in csv.DictReader(open(path)):
        if r["Type"] != "PASS" or r["Team"] not in ("Home", "Away"): continue
        f = lambda s: None if s in ("", "NaN") else float(s)
        out.append({"team": "A" if r["Team"] == "Home" else "B", "period": int(r["Period"]), "f0": int(r["Start Frame"]), "f1": int(r["End Frame"] or 0),
                    "x0": f(r["Start X"]), "y0": f(r["Start Y"]), "x1": f(r["End X"]), "y1": f(r["End Y"]), "sub": r["Subtype"]})
    return out


def kind(g, fwd=3.0): return "forward" if g > fwd else "backward" if g < -fwd else "sideways"


def truth_rows(P, per, home_right):
    rows = []
    for p in P:
        if p["period"] != per or p["x0"] is None or p["x1"] is None: continue
        sgn = 1 if (p["team"] == "A") == home_right else -1; g = (p["x1"] - p["x0"]) * L * sgn
        rows.append({**p, "gain_m": g, "kind": kind(g), "progressive": g >= 10.0})
    return rows


def ours_passes(fr, per, noisy, seed=0):
    import metricalab as ML
    got = {}; orig = AN.passes
    def cap(*a, **k):
        r = orig(*a, **k); got["ps"] = r[0]; return r
    AN.passes = cap
    try: o = ML.ours(fr, per, noisy, seed=seed)
    finally: AN.passes = orig
    return got["ps"], o["attack_right"]


def summary(rows, completed_only=False):
    out = {}
    for tm in ("A", "B"):
        r = [p for p in rows if p["team"] == tm and (p.get("completed", True) or not completed_only)]
        out[tm] = {"n": len(r), "forward_pct": round(100 * float(np.mean([p["kind"] == "forward" for p in r])), 1) if r else None,
                   "progressive": int(sum(1 for p in r if p["progressive"]))}
    return out


def match(T, ours, f0, tol_s=1.0):
    used = set(); pairs = []
    for t in T:
        best = None
        for j, u in enumerate(ours):
            if j in used or u["team"] != t["team"] or not u["completed"]: continue
            dt = abs(f0 + u["t"] * FPS - t["f0"])
            if dt <= tol_s * FPS and (best is None or dt < best[0]): best = (dt, j)
        if best is not None: used.add(best[1]); pairs.append((t, ours[best[1]]))
    kinds = sum(a["kind"] == b["kind"] for a, b in pairs)
    prog = [(a["progressive"], b["progressive"]) for a, b in pairs]
    err = [b["gain_m"] - a["gain_m"] for a, b in pairs]
    return {"n": len(pairs), "kind_agree": kinds, "prog_truth": sum(a for a, b in prog), "prog_ours": sum(b for a, b in prog),
            "prog_both": sum(a and b for a, b in prog), "gain_err_median_m": round(float(np.median(err)), 2) if err else None,
            "gain_abs_err_median_m": round(float(np.median(np.abs(err))), 2) if err else None}


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), noises=(False, "streak", "flicker"), seeds=(0, 1)):
    import metricalab as ML
    rows = []
    for g in games:
        gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
        fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}; Pz = load_passes(f"{gd}/{g}_RawEventsData.csv")
        for per in halves:
            hr = home_attacks_right(fr, per); T = truth_rows(Pz, per, hr); f0 = min(k for k in fr if fr[k][0] == per)
            for noisy in noises:
                for seed in (seeds if noisy else (0,)):
                    ps, ar = ours_passes(fr, per, noisy, seed)
                    row = {"game": g[-1], "half": per, "noise": str(noisy), "seed": seed, "dir_ok": ar["A"] == hr,
                           "truth": summary(T), "ours_all": summary(ps), "ours_completed": summary(ps, True), "matched": match(T, ps, f0)}
                    rows.append(row); print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-23c/metrica_progressive.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = main(a[0]); json.dump(rows, open(out, "w"), indent=1)
