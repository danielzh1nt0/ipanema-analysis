"""N0-24a: is the app's pass completion % right? stats.json teams[].pass_completion_pct = share of a team's passes
(analytics.passes: ball goes from one player's touch to the next player's touch) where the next player is a team-mate.
Also teams[].passes_per_sequence = passes / the team's exported sequences. Never checked on pro data before.

Truth (Metrica's labelled events): completed = PASS; a failed pass attempt = BALL LOST that is not a dribble lost
(THEFT / FORCED / HEAD-FORCED), not END HALF / OFFSIDE / REFEREE HIT. Two truth versions:
  'kept'  = PASS / (PASS + failed attempts)              (a pass to an opponent; what our rule can see)
  'all'   = also counts BALL OUT after a pass as failed  (the usual broadcast figure)
Truth sequences = metricalab.truth_sequences.
Ours = analytics.passes as the pipeline runs it (tools/metricalab.ours) on Metrica's positions, perfect ball and our
'streak' / 'flicker' ball noise (2 seeds each). Per attempt: truth failed attempts matched to our passes (same team,
our passer's last touch within 1 s of the event start) -> do we call it failed?

    PYTHONPATH=. python tools/n0_24a_completion.py <metrica sample-data/data dir> [--out results/qa/n0-24a/metrica_completion.json]
"""
import sys, os, json, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import analytics as AN

FPS = 25.0
KEEPS = (0, 1.0, 2.0, 3.0)
NOT_A_PASS = {"THEFT", "FORCED", "HEAD-FORCED", "END HALF", "OFFSIDE", "REFEREE HIT"}


def load_attempts(path):
    """-> [{team A/B, period, f0, ok (True = PASS), how (PASS / LOST / OUT)}]; BALL OUT only counts when it directly
    follows a PASS-like kick by the same team (no other on-ball event in between)"""
    ev = [r for r in csv.DictReader(open(path)) if r["Team"] in ("Home", "Away")]
    out = []
    for r in ev:
        tm = "A" if r["Team"] == "Home" else "B"; base = {"team": tm, "period": int(r["Period"]), "f0": int(r["Start Frame"]), "sub": r["Subtype"]}
        if r["Type"] == "PASS": out.append({**base, "ok": True, "how": "PASS"})
        elif r["Type"] == "BALL LOST" and r["Subtype"] not in NOT_A_PASS: out.append({**base, "ok": False, "how": "LOST"})
        elif r["Type"] == "BALL OUT" and "WOODWORK" not in r["Subtype"]: out.append({**base, "ok": False, "how": "OUT"})
    return out


def truth_pct(A, per, tm, with_out):
    r = [a for a in A if a["period"] == per and a["team"] == tm and (with_out or a["how"] != "OUT")]
    return (round(100 * sum(a["ok"] for a in r) / len(r), 1) if r else None), len(r)


def ours_pct(ps, tm):
    r = [p for p in ps if p["team"] == tm]
    return (round(100 * sum(p["completed"] for p in r) / len(r), 1) if r else None), len(r)


def match_attempts(A, ours, per, f0, tol_s=1.0):
    """each truth attempt (PASS / LOST) -> nearest unused our pass of the same team within tol -> (truth ok, ours completed)"""
    used = set(); pairs = []
    for a in sorted([a for a in A if a["period"] == per and a["how"] != "OUT"], key=lambda a: a["f0"]):
        best = None
        for j, u in enumerate(ours):
            if j in used or u["team"] != a["team"]: continue
            dt = abs(f0 + u["t"] * FPS - a["f0"])
            if dt <= tol_s * FPS and (best is None or dt < best[0]): best = (dt, j)
        if best is not None: used.add(best[1]); pairs.append((a["ok"], bool(ours[best[1]]["completed"])))
    n_fail = sum(1 for a in A if a["period"] == per and a["how"] == "LOST")
    return {"matched": len(pairs), "ok_ok": sum(a and b for a, b in pairs), "ok_called_failed": sum(a and not b for a, b in pairs),
            "failed_called_ok": sum((not a) and b for a, b in pairs), "failed_failed": sum((not a) and (not b) for a, b in pairs),
            "truth_failed_total": n_fail, "ours_failed_total": sum(1 for u in ours if not u["completed"])}


def drop_handbacks(ps, keep_s):
    """lab rule: a pass to an opponent whose receiver plays it straight back to the passer's team (his next pass is
    also failed) within keep_s is a ball flicker / duel, not two failed passes -> both dropped. keep_s 0 = off."""
    if not keep_s: return list(ps)
    ps = sorted(ps, key=lambda p: p["t"]); drop = set()
    for i, p in enumerate(ps):
        if p["completed"] or i in drop: continue
        j = next((j for j in range(i + 1, len(ps)) if ps[j]["from"] == p["to"] or ps[j]["team"] != p["team"]), None)
        if j is None: continue
        q = ps[j]
        if q["from"] == p["to"] and not q["completed"] and q["t"] - p["t"] < keep_s: drop |= {i, j}
    return [p for i, p in enumerate(ps) if i not in drop]


def ours_run(fr, per, noisy, seed):
    import metricalab as ML
    got = {}; orig = AN.passes
    def cap(*a, **k):
        r = orig(*a, **k); got["ps"] = r[0]; return r
    AN.passes = cap
    try: o = ML.ours(fr, per, noisy, seed=seed)
    finally: AN.passes = orig
    return got["ps"], o


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), noises=(False, "streak", "flicker"), seeds=(0, 1)):
    import metricalab as ML
    rows = []
    for g in games:
        gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
        fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}
        A = load_attempts(f"{gd}/{g}_RawEventsData.csv"); ev = ML.load_events(f"{gd}/{g}_RawEventsData.csv")
        for per in halves:
            f0 = min(k for k in fr if fr[k][0] == per); TS = ML.truth_sequences(ev, per)
            for noisy in noises:
                for seed in (seeds if noisy else (0,)):
                    ps0, o = ours_run(fr, per, noisy, seed)
                    for keep in KEEPS:
                        ps = drop_handbacks(ps0, keep)
                        oseq = {tm: sum(1 for s in o["sequence_starts"] if s[1] == tm) for tm in ("A", "B")}
                        row = {"game": g[-1], "half": per, "noise": str(noisy), "seed": seed, "handback_s": keep, "teams": {}}
                        for tm in ("A", "B"):
                            tk, nk = truth_pct(A, per, tm, False); ta, na = truth_pct(A, per, tm, True); op, on = ours_pct(ps, tm)
                            tseq = sum(1 for s in TS if s["team"] == tm); tpass = sum(1 for a in A if a["period"] == per and a["team"] == tm and a["ok"])
                            row["teams"][tm] = {"truth_kept_pct": tk, "truth_all_pct": ta, "ours_pct": op, "truth_attempts": nk, "truth_attempts_all": na, "ours_passes": on,
                                                "truth_passes_per_seq": round(tpass / tseq, 2) if tseq else None, "truth_sequences": tseq,
                                                "ours_passes_per_seq": round(on / oseq[tm], 2) if oseq[tm] else None, "ours_sequences": oseq[tm]}
                        row["matched"] = match_attempts(A, ps, per, f0)
                        rows.append(row); print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-24a/metrica_completion.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = main(a[0]); json.dump(rows, open(out, "w"), indent=1)
