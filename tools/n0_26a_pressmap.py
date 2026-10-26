"""N0-26a: are the app's pressing-over-time chart and pressure map right? Never checked on pro data before.

The app reads metrics.pressure_points[] (ipanema/metrics.pressure_points, every run): one point per second per carrier
while an opponent is within 2 m of him; 'carrier' = the player nearest the ball (<= 2.5 m, possession.carriers), no
possession state involved. The app counts the points per 5 minutes per pressing_team (two lines, 'Pressing over time',
lovable_fixes_2026-10-04 #10) and draws them as the pressure map.

Truth (Metrica events + tracking, 2 games x 2 halves): the same rule with the true carrier (n0_24d_ppda.truth_carrier:
the player of the latest on-ball event, the receiver after a pass has arrived, nobody while a pass is in flight) and the
true positions, only while the ball is in play (Metrica blanks it when out). So this measures what our ball / possession noise does to the chart and the map.
Ours = the pipeline (metricalab.ours: perfect ball, 'streak' and 'flicker' ball noise, 2 seeds).

Graded: count per team per half; per 5-min block per team (ratio, error, which team pressed more = the chart's story);
where (thirds in the pressing team's own direction, 12x8 grid misplacement); each point (same team, +-1 s, 5 m).
Rules: 'old' = the raw nearest-player carrier (the app's rule, still the default); 'new' = only frames where the counted
spell state gives the carrier's team the ball (option IPANEMA_PRESS_POINTS=spell); --lab adds 'lab_raw_state' (the raw state instead).

    PYTHONPATH=. python tools/n0_26a_pressmap.py <metrica sample-data/data dir> [--out results/qa/n0-26a/metrica_pressmap.json] [--lab] [--quick]
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P, metrics as M
import n0_24d_ppda as D

L, W, FPS = D.L, D.W, D.FPS
PRESS_R = 2.0
BLOCK_S = 300
GRID = (12, 8)


def truth_points(fr, per, car, ks):
    """the app's rule with the true carrier -> list of points (frame index from the half's start, like ours)"""
    out = []; last = {}; k0 = ks[0]
    for k in ks:
        c = car.get(k); f = fr.get(k)
        if c is None or f is None or f[2] is None: continue        # Metrica blanks the ball when it is out: no pressure on a dead ball
        cp = next(((x, y) for pid, x, y in f[1] if pid == c), None)
        if cp is None: continue
        ctm = "A" if c < 200 else "B"
        od = [np.hypot(x - cp[0], y - cp[1]) for pid, x, y in f[1] if (pid < 200) != (c < 200)]
        if not od or min(od) > PRESS_R: continue
        if k - last.get(c, -99) < FPS: continue
        last[c] = k
        out.append({"t": round((k - k0) / FPS, 2), "pressing_team": "B" if ctm == "A" else "A", "carrier_team": ctm, "x_m": round(cp[0], 1), "y_m": round(cp[1], 1)})
    return out


def ours_run(fr, per, noisy, seed):
    """metricalab.ours with the carriers / states captured -> (frames_, raw state, spell state, attack_right)"""
    import metricalab as ML
    got = {}
    o_car, o_ps, o_sp = P.carriers, P.pipeline_state, P.spell_state
    def car_(*a, **k): r = o_car(*a, **k); got["frames_"] = r[0]; return r
    def ps_(*a, **k): r = o_ps(*a, **k); got["state"] = r[0]; return r
    def sp_(*a, **k): r = o_sp(*a, **k); got["cstate"] = r; return r
    P.carriers, P.pipeline_state, P.spell_state = car_, ps_, sp_
    try: o = ML.ours(fr, per, noisy, seed=seed)
    finally: P.carriers, P.pipeline_state, P.spell_state = o_car, o_ps, o_sp
    return got["frames_"], np.asarray(got["state"]), np.asarray(got["cstate"]), o["attack_right"]


def gated(frames_, state):
    """lab: blank the carrier where the possession state does not give his team the ball"""
    out = []
    for k, f in enumerate(frames_):
        if f["carrier"] is not None and (state[k] > 1 or P.STATES[state[k]] != f["team"]): f = {**f, "carrier": None}
        out.append(f)
    return out


def blocks(pts, n_frames):
    nb = int(np.ceil(n_frames / FPS / BLOCK_S)); c = {"A": [0] * nb, "B": [0] * nb}
    for p in pts: c[p["pressing_team"]][min(nb - 1, int(p["t"] // BLOCK_S))] += 1
    return c


def thirds(pts, ar, tm):
    """share of the team's pressure points in its own / middle / attacking third (its own attack direction)"""
    s = [0, 0, 0]
    for p in pts:
        if p["pressing_team"] != tm: continue
        x = p["x_m"] if ar[tm] else L - p["x_m"]
        s[min(2, max(0, int(3 * x / L)))] += 1
    n = sum(s); return [round(100 * v / n, 1) for v in s] if n else None


def grid(pts, tm):
    g = np.zeros(GRID)
    for p in pts:
        if p["pressing_team"] != tm: continue
        i = min(GRID[0] - 1, max(0, int(p["x_m"] / L * GRID[0]))); j = min(GRID[1] - 1, max(0, int(p["y_m"] / W * GRID[1])))
        g[i, j] += 1
    return g / g.sum() if g.sum() else g


def match_points(ours, truth, dt=1.0, dm=5.0):
    """greedy one-to-one: same pressing team, |t| <= dt, <= dm metres -> matched count"""
    used = set(); m = 0
    by = sorted(range(len(truth)), key=lambda i: truth[i]["t"]); tt = np.array([truth[i]["t"] for i in by]) if by else np.zeros(0)
    for p in ours:
        lo, hi = np.searchsorted(tt, p["t"] - dt), np.searchsorted(tt, p["t"] + dt, side="right")
        best = None
        for jj in range(lo, hi):
            i = by[jj]; q = truth[i]
            if i in used or q["pressing_team"] != p["pressing_team"]: continue
            d = np.hypot(q["x_m"] - p["x_m"], q["y_m"] - p["y_m"])
            if d <= dm and (best is None or d < best[0]): best = (d, i)
        if best: used.add(best[1]); m += 1
    return m


def grade(ours, truth, ar_o, ar_t, n_frames):
    bo, bt = blocks(ours, n_frames), blocks(truth, n_frames)
    nb = len(bt["A"]); full = [b for b in range(nb) if (b + 1) * BLOCK_S * FPS <= n_frames + FPS * 60]   # blocks of >= 4 min
    top = [(bo["A"][b] > bo["B"][b]) == (bt["A"][b] > bt["B"][b]) for b in full if bt["A"][b] != bt["B"][b]]
    err = [abs(bo[t][b] - bt[t][b]) / bt[t][b] for t in "AB" for b in full if bt[t][b]]
    mt = match_points(ours, truth)
    res = {"blocks": len(full), "block_top_right": sum(top), "block_top_n": len(top), "block_err_median_pct": round(100 * float(np.median(err)), 1) if err else None,
           "block_counts_ours": bo, "block_counts_truth": bt, "points_ours": len(ours), "points_truth": len(truth),
           "matched": mt, "ours_real_pct": round(100 * mt / len(ours), 1) if ours else None, "truth_found_pct": round(100 * mt / len(truth), 1) if truth else None, "teams": {}}
    for tm in "AB":
        no = sum(p["pressing_team"] == tm for p in ours); nt = sum(p["pressing_team"] == tm for p in truth)
        go, gt = grid(ours, tm), grid(truth, tm)
        res["teams"][tm] = {"ours": no, "truth": nt, "ratio": round(no / nt, 2) if nt else None, "thirds_ours": thirds(ours, ar_o, tm), "thirds_truth": thirds(truth, ar_t, tm),
                            "grid_misplaced_pct": round(50 * float(np.abs(go - gt).sum()), 1)}
        a, b = res["teams"][tm]["thirds_ours"], res["teams"][tm]["thirds_truth"]
        res["teams"][tm]["thirds_off_pts"] = round(max(abs(x - y) for x, y in zip(a, b)), 1) if a and b else None
    no_, nt_ = res["teams"]["A"]["ours"], res["teams"]["A"]["truth"]
    res["more_pressing_right"] = (no_ > res["teams"]["B"]["ours"]) == (nt_ > res["teams"]["B"]["truth"])
    return res


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), noises=(False, "streak", "flicker"), seeds=(0, 1), lab=False):
    rows = []
    for g in games:
        fr, ev = D.load_game(d, g)
        for per in halves:
            ks = sorted(k for k in fr if fr[k][0] == per); ar_t = D.truth_direction(fr, per)
            car = D.truth_carrier(ev, per, ks[0], ks[-1]); tpts = truth_points(fr, per, car, ks)
            for noisy in noises:
                for seed in (seeds if noisy else (0,)):
                    frames_, state, cstate, ar = ours_run(fr, per, noisy, seed)
                    variants = {"old": M.pressure_points(None, frames_, FPS), "new": M.pressure_points(None, frames_, FPS, state=cstate)}   # new = the option
                    if lab:
                        variants["lab_raw_state"] = M.pressure_points(None, gated(frames_, state), FPS)
                        assert M.pressure_points(None, gated(frames_, cstate), FPS) == variants["new"]
                    for name, opts in variants.items():
                        row = {"game": g[-1], "half": per, "noise": str(noisy), "seed": seed, "rule": name, "direction_same": ar == ar_t, **grade(opts, tpts, ar, ar_t, len(ks))}
                        rows.append(row)
                        print(json.dumps({k: v for k, v in row.items() if not k.startswith("block_counts")}), flush=True)
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-26a/metrica_pressmap.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    kw = {"lab": "--lab" in a}
    if "--quick" in a: kw.update(games=("Sample_Game_1",), halves=(1,), noises=(False,))
    rows = main(a[0], **kw)
    json.dump(rows, open(out, "w"), indent=1)
