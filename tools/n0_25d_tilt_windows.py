"""N0-25d: is the app's 'possession over time' / 'field tilt over time' chart right? Never checked on pro data.

  metrics.tilt_windows (ipanema/metrics.py): 15-s windows, possession_A = A's share of the controlled frames (raw state,
      as run.py passes it), tilt_A = A's share of the frames controlled in a final third, control_s = seconds of control.
  The app (results/app/lovable_fixes_2026-10-04.md section 10) draws possession_A as a 5-min rolling mean of the windows
      (null windows skipped, every window counts the same) + 5-min bars 'SFK 58% · 0-5''; field tilt the same with tilt_A.

Truth (Metrica events + tracking, as tools/n0_24b_hightov.truth_tilt): per frame the team of the last on-ball event, dead
time (ball out -> restart) nobody's; ball spot from the tracking. Windows cut on the same frame grid as ours (from the
half's first frame).
Ours = the pipeline on Metrica's positions (tools/metricalab.ours via n0_24b_hightov.ours_run: perfect ball, streak and
flicker ball noise, 2 seeds).

Also a lab: the 5-min curve weighted by control_s (a window with 1 s of control no longer counts as much as one with 15 s).

    PYTHONPATH=. python tools/n0_25d_tilt_windows.py /tmp/metrica/data [--out results/qa/n0-25d/metrica_tilt_windows.json]
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import metrics as M

L, FPS, WIN_S, ROLL_S, BAR_S = 105.0, 25.0, 15, 150, 300
ON_BALL = ("PASS", "CARRY", "RECOVERY", "SHOT", "SET PIECE", "BALL LOST")


def is_foul(r):
    """a foul (FAULT RECEIVED, or a CHALLENGE whose subtype says FAULT) - play is dead until the next set piece"""
    return r["Type"] == "FAULT RECEIVED" or (r["Type"] == "CHALLENGE" and "FAULT" in (r.get("Subtype") or ""))


def truth_own(ev_rows, fr, per):
    """frame -> 'A' / 'B' (team of the last on-ball event; ball out or a foul -> restart = nobody's). n0_24b truth_tilt's rule
    plus fouls."""
    E = sorted([r for r in ev_rows if int(r["Period"]) == per], key=lambda r: int(r["Start Frame"]))
    ks = sorted(k for k in fr if fr[k][0] == per); f1 = ks[-1]
    own = {}; cur = None; dead = False
    for a, b in zip(E, E[1:] + [None]):
        ty = a["Type"]
        if ty == "BALL OUT" or is_foul(a): dead = True; cur = None   # a foul stops play until the free kick (n0_24b missed it:
        #   game 1 half 1 25:39-28:19, an injury after a foul, was all 'Away')
        elif ty in ON_BALL and a["Team"] in ("Home", "Away"): dead = False; cur = "A" if a["Team"] == "Home" else "B"
        end = int(b["Start Frame"]) if b is not None else f1 + 1
        if cur and not dead:
            for k in range(int(a["Start Frame"]), min(end, f1 + 1)): own[k] = cur
    return own


def truth_windows(own, fr, k0, n, ar):
    """the same windows as M.tilt_windows, from the truth. state index i = Metrica frame k0 + i"""
    STI = {"A": 0, "B": 1}; state = [2] * n; ballm = {}
    for i in range(n):
        k = k0 + i; t = own.get(k); b = fr.get(k, (0, 0, None))[2]
        if t is not None: state[i] = STI[t]
        if b is not None: ballm[i] = np.array(b)
    return M.tilt_windows(state, ballm, FPS, L, ar, window_s=WIN_S)


def rolling(tw, key, weighted=False):
    """the app's 5-min curve: mean of the non-null windows within +-150 s of each window (weighted: by control_s)"""
    t = np.array([w["t"] for w in tw]); v = np.array([np.nan if w[key] is None else w[key] for w in tw], float)
    wt = np.array([w["control_s"] for w in tw], float) if weighted else np.ones(len(tw))
    out = []
    for i in range(len(tw)):
        sel = (np.abs(t - t[i]) <= ROLL_S) & ~np.isnan(v) & (wt > 0)
        out.append(float(np.average(v[sel], weights=wt[sel])) if sel.any() else np.nan)
    return np.array(out)


def bars(tw, key, weighted=False):
    """5-min bars (0-5', 5-10' ...): mean of the windows in the block (the app averages windows, so unweighted)"""
    t = np.array([w["t"] for w in tw]); v = np.array([np.nan if w[key] is None else w[key] for w in tw], float)
    wt = np.array([w["control_s"] for w in tw], float) if weighted else np.ones(len(tw))
    out = []
    for b0 in np.arange(0, t.max() + 1e-9, BAR_S):
        sel = (t >= b0) & (t < b0 + BAR_S) & ~np.isnan(v) & (wt > 0)
        out.append(float(np.average(v[sel], weights=wt[sel])) if sel.any() else np.nan)
    return np.array(out)


def pooled_bars(tw):
    """the truth's own 5-min possession: pooled frames, not a mean of windows"""
    t = np.array([w["t"] for w in tw]); out = []
    for b0 in np.arange(0, t.max() + 1e-9, BAR_S):
        ws = [w for w in tw if b0 <= w["t"] < b0 + BAR_S and w["possession_A"] is not None]
        c = sum(w["control_s"] for w in ws)
        out.append(sum(w["possession_A"] * w["control_s"] for w in ws) / c if c else np.nan)
    return np.array(out)


def err_stats(a, b, margin=0.10):
    """a = ours, b = truth (0-1). -> median / p90 abs error in points, share both present, side agreement where the truth
    is clearly one-sided (>= margin from 50%)"""
    ok = ~np.isnan(a) & ~np.isnan(b); e = np.abs(a[ok] - b[ok]) * 100
    clear = ok & (np.abs(b - 0.5) >= margin)
    side = np.sign(a[clear] - 0.5) == np.sign(b[clear] - 0.5)
    return {"n": int(ok.sum()), "n_truth": int((~np.isnan(b)).sum()), "med_pts": round(float(np.median(e)), 1) if len(e) else None,
            "p90_pts": round(float(np.percentile(e, 90)), 1) if len(e) else None, "bias_pts": round(float(np.mean((a[ok] - b[ok]) * 100)), 1) if len(e) else None,
            "clear": int(clear.sum()), "side_right": int(side.sum()),
            "corr": round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 2) if ok.sum() > 2 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0 else None}


MIN_CONTROL_LAB = (1, 2, 3, 5)


def thin_null(ws, x):
    """copies of the windows with possession_A / tilt_A null when the window has < x s of control"""
    return [{**w, "possession_A": w["possession_A"] if w["control_s"] >= x else None, "tilt_A": w["tilt_A"] if w["control_s"] >= x else None} for w in ws]


def evaluate(ow, tw):
    """ours windows vs truth windows (same grid) -> the rows the README reports"""
    n = min(len(ow), len(tw)); ow, tw = ow[:n], tw[:n]
    arr = lambda ws, k: np.array([np.nan if w[k] is None else w[k] for w in ws], float)
    r = {"windows": n}
    r["poss_window"] = err_stats(arr(ow, "possession_A"), arr(tw, "possession_A"))
    r["poss_roll_app"] = err_stats(rolling(ow, "possession_A"), rolling(tw, "possession_A", weighted=True))
    r["poss_roll_weighted"] = err_stats(rolling(ow, "possession_A", weighted=True), rolling(tw, "possession_A", weighted=True))
    r["poss_bars_app"] = err_stats(bars(ow, "possession_A"), pooled_bars(tw))
    r["poss_bars_weighted"] = err_stats(bars(ow, "possession_A", weighted=True), pooled_bars(tw))
    for x in MIN_CONTROL_LAB:                                   # lab: a window with < x s of control written as null
        th = thin_null(ow, x)
        r[f"poss_roll_min{x:g}"] = err_stats(rolling(th, "possession_A"), rolling(tw, "possession_A", weighted=True))
        r[f"poss_bars_min{x:g}"] = err_stats(bars(th, "possession_A"), pooled_bars(tw))
        r[f"poss_window_min{x:g}"] = err_stats(arr(th, "possession_A"), arr(tw, "possession_A"))
    r["tilt_window"] = err_stats(arr(ow, "tilt_A"), arr(tw, "tilt_A"))
    r["tilt_roll_app"] = err_stats(rolling(ow, "tilt_A"), rolling(tw, "tilt_A"))
    r["control_s_ours_vs_truth"] = round(sum(w["control_s"] for w in ow) / max(1e-9, sum(w["control_s"] for w in tw)), 2)
    r["thin_windows_ours"] = int(sum(1 for w in ow if w["possession_A"] is not None and w["control_s"] < 3))
    return r


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), noises=(False, "streak", "flicker"), seeds=(0, 1), keep=None):
    import csv, metricalab as ML, n0_24b_hightov as HT
    rows = []
    for g in games:
        gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
        fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}
        ev = list(csv.DictReader(open(f"{gd}/{g}_RawEventsData.csv")))
        for per in halves:
            ks = sorted(k for k in fr if fr[k][0] == per); k0 = ks[0]; n = len(ks)
            assert ks[-1] - k0 + 1 == n, "frames of a half must be contiguous (ours index i = k0 + i)"
            ar_t = HT.truth_direction(fr, per); own = truth_own(ev, fr, per)
            tw = truth_windows(own, fr, k0, n, ar_t)
            for noisy in noises:
                for seed in (seeds if noisy else (0,)):
                    o = {}
                    _, _, ar = _ours(HT, fr, per, noisy, seed, o)
                    ow = M.tilt_windows(o["state"], o["ballm"], FPS, L, ar, window_s=WIN_S)
                    row = {"game": g[-1], "half": per, "noise": str(noisy), "seed": seed, "direction_same": ar == ar_t, **evaluate(ow, tw)}
                    if keep is not None: keep[(g[-1], per, str(noisy), seed)] = (ow, tw)
                    rows.append(row); print(json.dumps(row), flush=True)
    return rows


def _ours(HT, fr, per, noisy, seed, o):
    """run n0_24b's ours_run and capture the raw state + ball metres it saw (it keeps them in a local dict)"""
    from ipanema import possession as P
    o_ps, o_car = P.pipeline_state, P.carriers
    def ps_(*a, **k): r = o_ps(*a, **k); o["state"] = r[0]; return r
    def car_(*a, **k): r = o_car(*a, **k); o["ballm"] = r[1]; return r
    P.pipeline_state, P.carriers = ps_, car_
    try: return HT.ours_run(fr, per, noisy, seed)
    finally: P.pipeline_state, P.carriers = o_ps, o_car


if __name__ == "__main__":
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-25d/metrica_tilt_windows.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = main(a[0]); json.dump(rows, open(out, "w"), indent=1)
