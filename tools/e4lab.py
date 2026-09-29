"""E4 (29 Sep): stats with the new default possession model (possession_simple) vs the old one (viterbi), computed
offline the same way ipanema/run.py does after the ball pick (P.pipeline_state -> direction, sequences, restarts,
turnovers, lanes, passes, stats, metrics), on SFK-BP (5 min from 1200 s) and Reymersholm (5 min from 1500 s, no pitch
calibration: pixels scaled to a 106x64 'pitch', player height as the ruler). Free, local, ~1 min.
    PYTHONPATH=. python tools/e4lab.py  -> results/possession/e4_2026-09-29.json"""
import os, sys, json, gzip, glob, pickle, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, possession as P, analytics as AN, metrics as M, linecal as LC
q = lambda *a: None
PRESS_R, NEAR_R, CARRIER_R, LANE_HALF, MAX_LANE = 2.0, 5.0, 2.5, 1.5, 45.0


def load(f):
    o = pickle.load(open(f, "rb")); o = o[0] if isinstance(o, tuple) else o
    return {int(k): [(float(a), float(b), float(c)) for a, b, c in v] for k, v in o.items()}


def sfk():
    d = np.load("results/picker/SFKBP1109_s1200.npz", allow_pickle=True); n = int(d["n"]); fps = float(d["fps"])
    rows = LC.find_rows(".", "SFKBP1109_s1200"); cal = LC.calibration_for_clip(rows[2], n, fps, 1920, 1080, offset_s=1200, log=q)
    H, L, W = cal["H"], cal["L"], cal["W"]
    def to_m(Hk, px): v = np.linalg.inv(Hk) @ np.array([px[0], px[1], 1.0]); return v[:2] / v[2]
    R = json.load(gzip.open("results/qa/tracktest_gpu/rows_all.json.gz", "rt"))["rows"]["new, RF-DETR"]
    new = {int(k): [[pid, t, to_m(H[int(k)], px), np.array(px), None, False] for pid, t, px, fl in v if px is not None] for k, v in R.items()}
    per = {k: new.get(k, []) for k in range(n)}                             # {frame: rows}, as in the pipeline
    C = "results/volume/cache/SFKBP1109_s1200/"
    cd = BL.fuse_candidates(load(C + "ball_cands_clicks_round_20260928_0108.pkl"), load(C + "ball_cands_wasb_1790008894_t2x2_thr0.05.pkl"))
    ball = BL.bridge(BL.pick_v2(cd, H, L, W, per=per, fps=fps, log=q), fps)
    A = json.load(open("results/review/who_answers.json"))["moments"]
    return dict(name="SFK-BP (calibrated)", per=per, ball=ball, H=[H[k] for k in range(n)], L=L, W=W, fps=fps, boxh=None, key=A)


def reym():
    MATCH = "p15u-vs-reymersholm-2026-09-18"; C = f"results/volume/cache/{MATCH}_s1500_d300/"
    clk = load(C + "ball_cands_clicks_st3_fz0.pkl"); wasb = load(sorted(glob.glob(C + "ball_cands_wasb_*_t2x2_thr0.05.pkl"))[-1])
    R = json.load(gzip.open("results/kaggle/track_reym/rows_all.json.gz", "rt")); fps = float(R["fps"])
    rows = R["rows"]["new, RF-DETR"]; bh = R.get("box_h", {}).get("new, RF-DETR", {})
    L, W = 106.0, 64.0; S_ = np.array([[1920 / L, 0, 0], [0, 1080 / W, 0], [0, 0, 1.0]])
    n = max(max(int(k) for k in rows) + 1, max(clk) + 1, max(wasb) + 1)
    pd = {int(k): [[pid, t, np.array([px[0] * L / 1920, px[1] * W / 1080]), np.array(px), None, fl] for pid, t, px, fl in v if px is not None] for k, v in rows.items()}
    boxh = {int(k): [h for (pid, t, px, fl), h in zip(rows[k], v) if px is not None] for k, v in bh.items()}
    per = {k: pd.get(k, []) for k in range(n)}
    ball = BL.bridge(BL.pick_v2(BL.fuse_candidates(clk, wasb), {k: S_ for k in range(n)}, L, W, per=pd, fps=fps, log=q), fps)
    A = json.load(open("results/review/who_answers_reym.json"))["moments"]
    return dict(name="Reymersholm (no calibration, night)", per=per, ball=ball, H=[S_] * n, L=L, W=W, fps=fps, boxh=boxh, key=A)


def grade(state, key):
    ok = tot = 0
    for a in key:
        want = {"dark": 0, "white": 1, "light": 1, "loose": 2}.get(a["truth"])
        if want is None or a["frame"] >= len(state): continue
        tot += 1; ok += int(state[a["frame"]] == want)
    return [ok, tot]


def stats(c, mode):
    per, ball, H, L, W, fps = c["per"], c["ball"], c["H"], c["L"], c["W"], c["fps"]; n = len(per)
    frames_, ballm = P.carriers(per, ball, H, CARRIER_R, NEAR_R)
    state, bspeed, dstate, info = P.pipeline_state(per, ball, ballm, H, fps, L, W, mode=mode, boxh=c["boxh"], log=q)
    attack_right, conf = P.direction(state, ballm, log=q)
    if min(conf.values()) < 0.15: attack_right = P.direction_from_keepers(per, L, log=q) or P.direction_fallback(per, L, log=q)
    seqs = P.sequences(state, ballm, bspeed, fps, L, attack_right); rst = P.restarts(dstate, ballm, fps, L, W)
    tvs = P.turnovers(per, frames_, state, ballm, fps, attack_right, PRESS_R, NEAR_R, min_before_s=info["turnover_s"], min_after_s=info["turnover_s"])
    ln = AN.lanes(per, frames_, attack_right, LANE_HALF, MAX_LANE)
    ps, tracks = AN.passes(per, frames_, tvs, ln, attack_right, fps)
    sh = AN.shapes(per, L, fps)
    st = AN.stats(per, frames_, tvs, ps, tracks, state, fps, L, W, attack_right, PRESS_R, NEAR_R)
    mx = M.compute(state, ballm, bspeed, fps, L, W, attack_right, rst, ps, st["players"], tvs, sh, per=per, frames_=frames_)
    ctrl = int((state < 2).sum())
    return {"who_has_ball_right": grade(state, c["key"]),
            "possession_pct_dark_A": round(100 * int((state == 0).sum()) / max(1, ctrl)), "loose_pct": round(100 * int((state == 2).sum()) / n),
            "dead_pct": round(100 * int((np.asarray(dstate) == 3).sum()) / n), "restarts": len(rst), "sequences": len(seqs),
            "sequence_median_s": round(float(np.median([s["duration_s"] for s in seqs])), 1) if seqs else None,
            "turnovers": len(tvs), "turnover_rule_s": info["turnover_s"], "passes": len(ps), "passes_completed": sum(p["completed"] for p in ps),
            "shots_detected": {t: sum(1 for s in mx["shots"] if s["team"] == t) for t in ("A", "B")},
            "field_tilt": {t: mx["field"][t]["field_tilt_pct"] for t in ("A", "B")}, "high_turnovers": mx["high_turnover_counts"],
            "state_counts": dict(sorted(collections.Counter(np.asarray(state).tolist()).items()))}


if __name__ == "__main__":
    out = {}
    for c in (sfk(), reym()):
        out[c["name"]] = {m: stats(c, m) for m in ("viterbi", "simple")}
        # the old pipeline rule for lost balls (3 s + 3 s) on the new state, for comparison
        for m in ("simple",):
            per, ball, H = c["per"], c["ball"], c["H"]; fr, bm = P.carriers(per, ball, H, CARRIER_R, NEAR_R)
            s, bs, ds, _ = P.pipeline_state(per, ball, bm, H, c["fps"], c["L"], c["W"], mode=m, boxh=c["boxh"], log=q)
            ar, _ = P.direction(s, bm, log=q)
            out[c["name"]]["simple"]["turnovers_with_3s_rule"] = len(P.turnovers(per, fr, s, bm, c["fps"], ar, PRESS_R, NEAR_R))
        print(c["name"])
        for m, r in out[c["name"]].items(): print(" ", m, json.dumps(r))
    os.makedirs("results/possession", exist_ok=True)
    json.dump(out, open("results/possession/e4_2026-09-29.json", "w"), indent=1)
