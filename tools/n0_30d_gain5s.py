"""N0-30d: is the 'X m in 5 s' on every 'won the ball' event right? (match_data events turnover_won subtitle +
payload.gain_5s_m = how far the ball moved towards the opponent goal in the 5 s after the win; P.turnovers). Never checked.

Truth = Metrica pro data: each labelled BALL LOST -> the winner's RECOVERY / first on-ball event (tools/n0_23a_transition),
gain = the TRUE ball's x 5 s after that frame minus at that frame, towards the winner's attacking goal. Ours = the pipeline's
kept turnovers (after P.confirm_losses) run on Metrica positions with a perfect ball and with our 'streak' / 'flicker'
ball noise (tools/metricalab.ours). Matched one by one (same winner, win within 2 s). Also: the same 5-s gain measured with
the TRUE ball from OUR win frame ('timing only') to split timing error from ball noise, and 'always 0 m' as a baseline.

    PYTHONPATH=. python tools/n0_30d_gain5s.py /tmp/metrica/data [--out results/qa/n0-30d/metrica_gain5s.json]
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

L, FPS = 105.0, 25.0
WIN_S = 5.0


def ball_at(fr, k, tol=12):
    """true ball (x, y) at Metrica frame k, else the nearest frame within tol frames that has one; None if none"""
    for d in range(tol + 1):
        for kk in ((k,) if d == 0 else (k - d, k + d)):
            v = fr.get(kk)
            if v is not None and v[2] is not None: return np.array(v[2])
    return None


def gain(fr, k, right, win_s=WIN_S, last=True):
    """metres the true ball moves towards the winner's attacking goal from frame k to k + win_s; last=True: when Metrica
    has no ball at the end (out of play), the last true ball inside the window (where it left), like the N0-30d rule"""
    a, b = ball_at(fr, k), ball_at(fr, k + int(win_s * FPS))
    if b is None and last and a is not None:
        b = next((np.array(fr[j][2]) for j in range(k + int(win_s * FPS), k + int(FPS), -1) if fr.get(j) and fr[j][2] is not None), None)
    if a is None or b is None: return None
    return round(float((b[0] - a[0]) * (1 if right else -1)), 1)


def match_pairs(T, kept, f0, tol_s=2.0):
    """truth wins matched to our kept turnovers (same winner, win frames within tol_s, nearest first) -> [(t, u)]"""
    used = set(); out = []
    for t in T:
        best = None
        for j, u in enumerate(kept):
            if j in used or u["won_by"] != t["won_by"]: continue
            dt = abs(f0 + u["frame"] - t["f_won"])
            if dt <= tol_s * FPS and (best is None or dt < best[0]): best = (dt, j)
        if best is None: continue
        used.add(best[1]); out.append((t, kept[best[1]]))
    return out


def score(pairs_xy):
    """[(truth, ours)] with both not None -> error figures"""
    v = [(a, b) for a, b in pairs_xy if a is not None and b is not None]
    if not v: return {"n": 0}
    a = np.array([x for x, _ in v]); b = np.array([y for _, y in v]); e = np.abs(a - b)
    sig = lambda x: np.where(x >= 5, 1, np.where(x <= -5, -1, 0))   # forward / about level / back, like a coach would read it
    return {"n": len(v), "mae_m": round(float(e.mean()), 1), "median_err_m": round(float(np.median(e)), 1),
            "within_5m_pct": round(100 * float((e <= 5).mean())), "within_10m_pct": round(100 * float((e <= 10).mean())),
            "dir_agree_pct": round(100 * float((sig(a) == sig(b)).mean())),
            "corr": round(float(np.corrcoef(a, b)[0, 1]), 2) if len(v) > 2 and a.std() > 0 and b.std() > 0 else None,
            "zero_mae_m": round(float(np.abs(a).mean()), 1), "truth_mean_m": round(float(a.mean()), 1), "ours_mean_m": round(float(b.mean()), 1)}


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), noises=(False, "streak", "flicker"), seeds=(0, 1), modes=("old", "new")):
    import metricalab as ML, n0_23a_transition as TR
    from ipanema import possession as PO
    rows = []; pairs_all = []
    for g in games:
        gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
        fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}; ev = TR.load_events_xy(f"{gd}/{g}_RawEventsData.csv")
        for per in halves:
            hr = TR.home_attacks_right(fr, per); T = TR.truth_transitions(ev, per, hr); f0 = min(k for k in fr if fr[k][0] == per)
            right = {"A": hr, "B": not hr}
            for t in T: t["gain_true"] = gain(fr, t["f_won"], right[t["won_by"]])
            for noisy, seed, mode in [(nz, sd, md) for nz in noises for sd in (seeds if nz else (0,)) for md in modes]:
                if True:
                    PO.GAIN_ONPITCH = mode == "new"
                    o = TR.ours_kept(fr, per, noisy, seed); kept = o["kept"]
                    P = match_pairs(T, kept, f0)
                    ours_vs = [(t["gain_true"], u["gain_5s_m"]) for t, u in P]
                    timing = [(t["gain_true"], gain(fr, f0 + u["frame"], right[u["won_by"]])) for t, u in P]   # true ball, our win frame
                    noise_only = [(gain(fr, f0 + u["frame"], right[u["won_by"]]), u["gain_5s_m"]) for t, u in P]  # same frame, our ball
                    all_ours = [(gain(fr, f0 + u["frame"], right[u["won_by"]]), u["gain_5s_m"]) for u in kept]       # every kept win
                    row = {"game": g[-1], "half": per, "noise": str(noisy), "seed": seed, "mode": mode, "over_40m": sum(abs(u["gain_5s_m"]) > 40 for u in kept if u["gain_5s_m"] is not None), "truth_wins": len(T),
                           "truth_with_gain": sum(t["gain_true"] is not None for t in T), "ours_wins": len(kept),
                           "ours_with_gain_pct": round(100 * np.mean([u["gain_5s_m"] is not None for u in kept])) if kept else None,
                           "matched": len(P), "ours_vs_truth": score(ours_vs), "timing_only": score(timing),
                           "ball_only": score(noise_only), "all_kept_vs_true_ball": score(all_ours)}
                    rows.append(row); print(json.dumps(row), flush=True)
                    pairs_all += [{"game": g[-1], "half": per, "noise": str(noisy), "seed": seed, "mode": mode, "won_by": u["won_by"], "t_won": u["t_won"],
                                   "truth": t["gain_true"], "ours": u["gain_5s_m"], "true_ball_our_frame": b} for (t, u), (_, b) in zip(P, timing)]
    return rows, pairs_all


def summarize(rows):
    """pool the halves per noise: weighted by n"""
    out = {}
    for nz in sorted({(r["noise"], r.get("mode", "old")) for r in rows}):
        R = [r for r in rows if (r["noise"], r.get("mode", "old")) == nz]; s = {}; nz = "/".join(nz)
        for key in ("ours_vs_truth", "timing_only", "ball_only", "all_kept_vs_true_ball"):
            n = sum(r[key].get("n", 0) for r in R)
            w = lambda f: round(sum(r[key][f] * r[key]["n"] for r in R if r[key].get("n")) / n, 1) if n else None
            s[key] = {"n": n, **{f: w(f) for f in ("mae_m", "median_err_m", "within_5m_pct", "within_10m_pct", "dir_agree_pct", "zero_mae_m", "truth_mean_m", "ours_mean_m")}}
        s["matched"] = sum(r["matched"] for r in R); s["truth_wins"] = sum(r["truth_wins"] for r in R); s["ours_wins"] = sum(r["ours_wins"] for r in R)
        s["over_40m"] = sum(r.get("over_40m", 0) for r in R); s["ours_with_gain_pct"] = round(float(np.mean([r["ours_with_gain_pct"] for r in R if r["ours_with_gain_pct"] is not None])))
        out[nz] = s
    return out


if __name__ == "__main__":
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-30d/metrica_gain5s.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows, pairs = main(a[0]); json.dump({"rows": rows, "summary": summarize(rows), "pairs": pairs}, open(out, "w"), indent=1)
    print(json.dumps(summarize(rows), indent=1))
