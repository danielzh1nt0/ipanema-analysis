"""N0-23b: is the app's counter-press figure right? stats.json teams[].time_to_press_median_s ("time until the team that
lost the ball gets a player within 2 m of the new carrier"), pressed_within_2s_pct and near_at_2s_median (losers within
5 m of the carrier 2 s after the win). Truth = Metrica's labelled BALL LOST -> the winner's RECOVERY (or first on-ball
event) + Metrica's full, true positions: the carrier is the winner's player nearest the true ball (<= 2.5 m, the app's
rule), pressure = nearest loser to him, until the winner's chain ends or 8 s. Ours = the pipeline's turnovers after
P.confirm_losses (what stats() gets) on Metrica's positions, perfect ball / our 'streak' (players thinned to ~65%) /
'flicker' ball noise (tools/metricalab.ours).

    PYTHONPATH=. python tools/n0_23b_press.py /tmp/metrica/data [--out results/qa/n0-23b/metrica_press.json]
    ... --view   players seen through our app clips' real camera views + detection gaps (perfect / flicker ball)
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P
from n0_23a_transition import load_events_xy, ours_kept, ONBALL

FPS = 25.0


def truth_press(ev, fr, per, press_r=2.0, near_r=5.0, carrier_r=2.5, look_s=8.0, recov_s=5.0):
    E = sorted([e for e in ev if e["period"] == per], key=lambda e: (e["f0"], e["f1"])); out = []
    for i, e in enumerate(E):
        if e["type"] != "BALL LOST" or e["team"] not in ("Home", "Away"): continue
        wt = "Away" if e["team"] == "Home" else "Home"; later = [q for q in E[i + 1:] if q["f0"] >= e["f0"]]
        win = next((q for q in later if q["team"] == wt and q["type"] == "RECOVERY" and q["f0"] - e["f0"] <= recov_s * FPS), None)
        if win is None: win = next((q for q in later if q["team"] == wt and q["type"] in ONBALL), None)
        if win is None: continue
        end = win["f0"] + int(look_s * FPS)
        for q in later:                                              # the winner's chain ends: other team acts / ball out
            if q["f0"] <= win["f0"]: continue
            if q["type"] == "BALL OUT" or (q["team"] == e["team"] and q["type"] in ("RECOVERY", "PASS", "CARRY", "SHOT", "SET PIECE", "CHALLENGE")):
                end = min(end, q["f0"]); break
        win_home = wt == "Home"; ttp = None; near2 = None
        for k in range(win["f0"], end + 1):
            if k not in fr: continue
            _, ps, b = fr[k]
            if b is None: continue
            mine = [(np.hypot(x - b[0], y - b[1]), x, y) for pid, x, y in ps if (pid < 200) == win_home]
            if not mine: continue
            d, cx, cy = min(mine)
            if d > carrier_r: continue
            od = [np.hypot(x - cx, y - cy) for pid, x, y in ps if (pid < 200) != win_home]
            if k == win["f0"] + int(2 * FPS): near2 = int(sum(x <= near_r for x in od))
            if ttp is None and od and min(od) <= press_r: ttp = round((k - win["f0"]) / FPS, 2)
        out.append({"f_lost": e["f0"], "f_won": win["f0"], "won_by": "A" if win_home else "B", "time_to_press": ttp, "near_at_2s": near2})
    return out


def summ(rows):
    t = [r["time_to_press"] for r in rows if r["time_to_press"] is not None]
    n2 = [r["near_at_2s"] for r in rows if r["near_at_2s"] is not None]
    w2 = [(r["time_to_press"] is not None and r["time_to_press"] <= 2) for r in rows]
    return {"n": len(rows), "median_s": round(float(np.median(t)), 2) if t else None,
            "within2_pct": round(100 * float(np.mean(w2)), 1) if w2 else None,
            "near2_median": float(np.median(n2)) if n2 else None, "pressed_ever_pct": round(100 * len(t) / max(1, len(rows)), 1)}


def matched(T, kept, f0, tol_s=2.0):
    used = set(); rows = []
    for t in T:
        best = None
        for j, u in enumerate(kept):
            if j in used or u["won_by"] != t["won_by"]: continue
            dt = abs(f0 + u["frame"] - t["f_won"])
            if dt <= tol_s * FPS and (best is None or dt < best[0]): best = (dt, j)
        if best is None: continue
        used.add(best[1]); u = kept[best[1]]
        rows.append({"truth": t["time_to_press"], "ours": u["time_to_press"], "dt_won_s": round((f0 + u["frame"] - t["f_won"]) / FPS, 2),
                     "dt_lost_s": round((f0 + u["frame_lost"] - t["f_lost"]) / FPS, 2), "truth_near2": t["near_at_2s"], "ours_near2": u["near_at_2s"]})
    a = [(r["truth"] is not None and r["truth"] <= 2) for r in rows]; b = [(r["ours"] is not None and r["ours"] <= 2) for r in rows]
    both = [(r["truth"], r["ours"]) for r in rows if r["truth"] is not None and r["ours"] is not None]
    return {"n": len(rows), "agree_w2": sum(x == y for x, y in zip(a, b)), "truth_w2": sum(a), "ours_w2": sum(b),
            "median_err_s": round(float(np.median([o - t for t, o in both])), 2) if both else None,
            "median_dt_won_s": round(float(np.median([r["dt_won_s"] for r in rows])), 2) if rows else None, "rows": rows}


CLIPS = ("results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl", "results/volume/cache/p15u-vs-aik-2026-09-21-bd09_s2520/picker_inputs.pkl")


def view_frames(fr, per, views, Lv, Wv, runs, gaps, jit, seed=0, every=15, Lm=105.0, Wm=68.0):
    """Metrica seen like our follow-cam: a real camera view from an app clip slid onto Metrica's ball, players outside it
    dropped, plus our clips' detection gaps (tools/n0_22e_shape_view.detection_plan) and position jitter. Ball untouched."""
    import n0_22e_shape_view as SV
    rng = np.random.default_rng(seed); ks = sorted(k for k in fr if fr[k][0] == per)
    plan = {i: SV.detection_plan(len(ks), FPS, rng, runs, gaps) for i in sorted({i for k in ks for i, _, _ in fr[k][1]})}
    out = {}; seen = tot = 0
    for j, k in enumerate(ks):
        p, ps, b = fr[k]
        if b is None: out[k] = (p, [], b); continue
        m, cx = views[(j // every) % len(views)]; inside = SV.place(m, cx, b[0], Lv, Wv, Lm, Wm); keep = []
        for i, x, y in ps:
            tot += 1
            if plan[i][j] >= 0 and inside(np.array([x]), np.array([y]))[0]:
                e = jit[rng.integers(0, len(jit))]; keep.append((i, x + e[0], y + e[1])); seen += 1
        out[k] = (p, keep, b)
    return out, round(seen / max(1, tot), 3)


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), noises=(False, "streak", "flicker"), seeds=(0, 1), view=False):
    import metricalab as ML
    if view:
        import n0_22e_shape_view as SV, n0_22c_metrica as C
        runs, gaps, jit = C.our_breaks(list(CLIPS)); V = [SV.view_masks(c) for c in CLIPS]
    out = []
    for g in games:
        gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
        fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}; ev = load_events_xy(f"{gd}/{g}_RawEventsData.csv")
        for per in halves:
            T = truth_press(ev, fr, per); f0 = min(k for k in fr if fr[k][0] == per)
            for noisy in noises:
                for seed in (seeds if noisy or view else (0,)):
                    fv, share = (fr, 1.0)
                    if view:
                        v, Lv, Wv = V[seed % len(V)]; fv, share = view_frames(fr, per, v, Lv, Wv, runs, gaps, jit, seed)
                    o = ours_kept(fv, per, noisy, seed); m = matched(T, o["kept"], f0)
                    row = {"game": g[-1], "half": per, "noise": str(noisy) + ("+view" if view else ""), "seed": seed, "seen_share": share, "truth": summ(T), "ours": summ(o["kept"]),
                           "matched": {k: v for k, v in m.items() if k != "rows"}, "pairs": m["rows"]}
                    out.append(row); print(json.dumps({k: v for k, v in row.items() if k != "pairs"}), flush=True)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-23b/metrica_press.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(main(a[0], noises=(False, "flicker") if "--view" in a else (False, "streak", "flicker"), view="--view" in a), open(out, "w"), indent=1)
