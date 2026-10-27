"""N0-27a: is the app's 'A better pass was open' right? (Insights card + stats.json teams[] / players[] better_option_count,
events of type better_option, passes[].better_option / best_to / quality). Never checked before.

There is no human label for 'a better pass was open', so the reference is OUR OWN RULE (analytics.best_option) fed
with perfect inputs: Metrica's labelled PASS events, every one of the 22 players at the pass's start frame, the
receiver = the passer's team-mate nearest the ball at the pass's end frame. That says what the rule means on real
football (how often it calls a pro's pass second best) and how far our inputs move it.
Ours = analytics.passes as the pipeline runs it (tools/metricalab.ours: possession state, carriers, turnovers) with
analytics.lanes like ipanema/run.py, on Metrica's positions: perfect ball / 'flicker' ball noise, all players or
players seen through our app clips' real camera views + detection gaps (tools/n0_23b_press.view_frames).
Per pass: ours matched to a truth pass (same team, completed, |t| <= 1 s) -> flag agreement.

    PYTHONPATH=. python tools/n0_27a_better_option.py /tmp/metrica/src/data [--out results/qa/n0-27a/metrica_better.json]
    PYTHONPATH=. python tools/n0_27a_better_option.py --dry      (stand-in game, for the test suite)
"""
import sys, os, json, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import analytics as AN

L, W, FPS = 105.0, 68.0, 25.0
CLIPS = ("results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl", "results/volume/cache/p15u-vs-aik-2026-09-21-bd09_s2520/picker_inputs.pkl")


def load_passes(path):
    out = []
    for r in csv.DictReader(open(path)):
        if r["Type"] != "PASS" or r["Team"] not in ("Home", "Away"): continue
        f = lambda s: None if s in ("", "NaN") else float(s)
        out.append({"team": "A" if r["Team"] == "Home" else "B", "period": int(r["Period"]), "f0": int(r["Start Frame"]), "f1": int(r["End Frame"] or 0),
                    "x1": f(r["End X"]), "y1": f(r["End Y"])})
    return out


def rows_of(ps):
    return [[pid, "A" if pid < 200 else "B", np.array([x, y], float), None, None, False] for pid, x, y in ps]


def truth_rows(Pz, fr, per, home_right, max_lane=45.0, lane_half=1.5):
    """the rule on perfect inputs, one row per real pass it can value (passer and receiver found, receiver in reach)"""
    out = []
    for p in Pz:
        if p["period"] != per or p["f0"] not in fr or p["f1"] not in fr or fr[p["f0"]][2] is None or p["x1"] is None: continue
        _, ps, b = fr[p["f0"]]; mine = [(np.hypot(x - b[0], y - b[1]), pid) for pid, x, y in ps if (pid < 200) == (p["team"] == "A")]
        if not mine: continue
        d, me = min(mine)
        if d > 2.5: continue
        e = np.array([p["x1"] * L, p["y1"] * W]); _, ps1, _ = fr[p["f1"]]
        rec = [(np.hypot(x - e[0], y - e[1]), pid) for pid, x, y in ps1 if (pid < 200) == (p["team"] == "A") and pid != me]
        if not rec: continue
        to = min(rec)[1]; rows = rows_of(ps); sgn = 1 if (p["team"] == "A") == home_right else -1
        ln = AN.lanes([rows], [{"carrier": me}], {p["team"]: sgn == 1, ("B" if p["team"] == "A" else "A"): sgn != 1}, lane_half, max_lane)[0]
        bo = AN.best_option(rows, me, p["team"], sgn, ln, to)
        if not bo or bo["played_value"] is None: continue
        out.append({"f0": p["f0"], "team": p["team"], "from": me, "to": to, **bo})
    return out


def ours_passes(fr, per, noisy, seed=0):
    """the pipeline's passes with lanes, exactly as run.py builds them"""
    import metricalab as ML
    got = {}; orig = AN.passes
    def cap(per_, frames_, tvs, lanes_, ar, fps, **k):
        ln = AN.lanes(per_, frames_, ar, 1.5, 45.0); r = orig(per_, frames_, tvs, ln, ar, fps, **k); got["ps"] = r[0]; return r
    AN.passes = cap
    try: o = ML.ours(fr, per, noisy, seed=seed)
    finally: AN.passes = orig
    return got["ps"], o["attack_right"]


def summ(rows, team=None):
    r = [x for x in rows if team is None or x["team"] == team]; v = [x for x in r if x.get("played_value") is not None]
    return {"n": len(r), "valued": len(v), "better": int(sum(bool(x.get("better_option")) for x in r)),
            "better_pct_of_valued": round(100 * sum(bool(x.get("better_option")) for x in v) / max(1, len(v)), 1)}


def match(T, ours, f0, tol_s=1.0):
    used = set(); pairs = []
    for t in T:
        best = None
        for j, u in enumerate(ours):
            if j in used or u["team"] != t["team"] or not u["completed"] or u.get("played_value") is None: continue
            dt = abs(f0 + u["t"] * FPS - t["f0"])
            if dt <= tol_s * FPS and (best is None or dt < best[0]): best = (dt, j)
        if best is not None: used.add(best[1]); pairs.append((t, ours[best[1]]))
    tb = [bool(a["better_option"]) for a, b in pairs]; ob = [bool(b["better_option"]) for a, b in pairs]
    both = sum(x and y for x, y in zip(tb, ob)); same_best = sum(1 for a, b in pairs if a["better_option"] and b["better_option"] and a["best_to"] == b["best_to"])
    return {"n": len(pairs), "agree": int(sum(x == y for x, y in zip(tb, ob))), "truth_yes": int(sum(tb)), "ours_yes": int(sum(ob)), "both_yes": int(both),
            "ours_yes_real_pct": round(100 * both / max(1, sum(ob)), 1), "truth_yes_found_pct": round(100 * both / max(1, sum(tb)), 1), "same_better_mate": int(same_best),
            "same_receiver": int(sum(a["to"] == b["to"] for a, b in pairs))}


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), variants=((False, False), ("flicker", False), (False, True), ("flicker", True)), seeds=(0, 1)):
    import metricalab as ML, n0_23b_press as PR
    from n0_23a_transition import home_attacks_right
    runs = gaps = jit = V = None
    if any(v for _, v in variants):
        import n0_22e_shape_view as SV, n0_22c_metrica as C
        runs, gaps, jit = C.our_breaks(list(CLIPS)); V = [SV.view_masks(c) for c in CLIPS]
    out = []
    for g in games:
        gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
        fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}; Pz = load_passes(f"{gd}/{g}_RawEventsData.csv")
        for per in halves:
            hr = home_attacks_right(fr, per); T = truth_rows(Pz, fr, per, hr); f0 = min(k for k in fr if fr[k][0] == per)
            for noisy, view in variants:
                for seed in (seeds if noisy or view else (0,)):
                    fv, share = fr, 1.0
                    if view:
                        v, Lv, Wv = V[seed % len(V)]; fv, share = PR.view_frames(fr, per, v, Lv, Wv, runs, gaps, jit, seed)
                    ps, ar = ours_passes(fv, per, noisy, seed)
                    row = {"game": g[-1], "half": per, "noise": str(noisy) + ("+view" if view else ""), "seed": seed, "seen_share": share, "dir_ok": ar["A"] == hr,
                           "truth": {tm: summ(T, tm) for tm in ("A", "B")}, "ours": {tm: summ(ps, tm) for tm in ("A", "B")}, "matched": match(T, ps, f0)}
                    out.append(row); print(json.dumps(row), flush=True)
    return out


def dry():
    """stand-in: a 3-v-3 pass where the passer ignores a free team-mate 20 m further forward -> truth row says better"""
    fr = {}
    for k in range(0, 40):
        b = (30.0 + (0 if k < 10 else min(1, (k - 10) / 10) * 5), 30.0)
        fr[k] = (1, [(101, 30.0, 30.0), (102, 35.0, 30.0), (103, 55.0, 40.0), (201, 45.0, 20.0), (202, 60.0, 10.0), (203, 100.0, 34.0)], b)
    Pz = [{"team": "A", "period": 1, "f0": 10, "f1": 20, "x1": 35.0 / L, "y1": 30.0 / W}]
    T = truth_rows(Pz, fr, 1, True)
    assert len(T) == 1 and T[0]["to"] == 102 and T[0]["best_to"] == 103 and T[0]["better_option"], T
    return T


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--dry" in a: print(json.dumps(dry())); sys.exit(0)
    out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-27a/metrica_better.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(main(a[0]), open(out, "w"), indent=1)
