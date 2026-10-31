"""N0-31a: is the app's ball grade right? (summary.ball_grade -> possession_ok / events_ok, written by ipanema/run.py;
the app hides possession, losses, pass completion, field tilt and the pressing screen when possession_ok is false
(Vallentuna today) and shows 'Beta - low confidence' when events_ok is false (every match today)). Never checked.

The grade (run.py):
  near_frac   = share of ball frames with a VISIBLE player within 4 m of the ball
  accuracy    = checked frames where our ball is within 30 px of a click (needs >= 10 checked frames)
  possession_ok = accuracy >= 0.6, or with no checked frames: near_frac >= 0.85 and ball in > 40% of match frames
  events_ok     = accuracy >= 0.6 and near_frac >= 0.85

Test bed = Metrica's 2 free pro games (4 halves), run through the pipeline's own steps (tools/metricalab.ours, like
the nightly). Balls: perfect; our clips' kinds of noise ('flicker', 'streak', 'veo' = 65% of players seen, 25% of balls
missing, 10% wrong picks); and two made-up BAD balls the grade should catch:
  'garbage' - a random spot on the pitch every 2 s (nothing to do with play)
  'sticky'  - the ball dot sits on one random player (any team) for 2 s, then another: always 'near a player',
              but the team on the ball is a coin flip.
Per run: near_frac exactly like run.py, the same for the TRUE ball (what a perfect ball scores), an accuracy proxy
(40 random frames with a true ball: ours within 2 m, a miss counts wrong - like 30 px on a Veo frame, roughly),
the possession error vs the frame truth (tools/n0_29f_possession, Metrica's own 2-s+ ball gaps left out), the pass
count error, and the grade's verdicts.

    PYTHONPATH=. python tools/n0_31a_ballgrade.py /tmp/metrica/data --game Sample_Game_1 --out a.json
    PYTHONPATH=. python tools/n0_31a_ballgrade.py --merge a.json b.json --out results/qa/n0-31a/metrica_ballgrade.json
    PYTHONPATH=. python tools/n0_31a_ballgrade.py --dry
"""
import sys, os, json, random, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P

FPS, L, W = 25.0, 105.0, 68.0
NEAR_M = 4.0            # run.py
HIT_M = 2.0             # accuracy proxy: ~30 px on a Veo frame
N_CHECK = 40            # app clips have 34-40 checked frames


def near_frac(perd, ballm):
    """run.py's rule: share of ball frames with a visible player within NEAR_M of the ball (frames with nobody seen = not near)"""
    if not ballm: return 0.0
    near = sum(1 for k, b in ballm.items() if perd.get(k) and min(np.linalg.norm(np.asarray(r[2]) - np.asarray(b)) for r in perd[k]) < NEAR_M)
    return near / len(ballm)


def frame_agree(st, own, k0, keep):
    """share of frames (truth has a team, ours says A or B, outside Metrica's ball gaps) where ours names the truth's team"""
    st = np.asarray(st); hit = tot = 0
    for i, v in enumerate(st):
        t = own.get(k0 + i)
        if not keep[i] or t is None or v not in (0, 1): continue
        tot += 1; hit += int("AB"[int(v)] == t)
    return round(hit / tot, 3) if tot else None


def gate(acc, nf, frames_pct):
    """run.py's verdicts (copied rule; tests/test_n0_31a_ballgrade.py checks it against run.py's source)"""
    pos = (acc >= 0.6) if acc is not None else (nf >= 0.85 and frames_pct > 0.4)
    ev = acc is not None and acc >= 0.6 and nf >= 0.85
    return bool(pos), bool(ev)


def accuracy(ballm, true_ball, ks_idx, seed, n=N_CHECK):
    """ks_idx = [(i, true (x, y))] frames with a true ball; n random ones; ours within HIT_M = right, missing = wrong"""
    rng = random.Random(1000 + seed); pick = rng.sample(ks_idx, min(n, len(ks_idx)))
    ok = sum(1 for i, t in pick if i in ballm and np.hypot(ballm[i][0] - t[0], ballm[i][1] - t[1]) <= HIT_M)
    full = sum(1 for i, t in ks_idx if i in ballm and np.hypot(ballm[i][0] - t[0], ballm[i][1] - t[1]) <= HIT_M) / max(1, len(ks_idx))
    return ok / max(1, len(pick)), full


def bad_ball(fr, per, kind, seed, hold_s=2.0):
    """copy of fr with the ball of half `per` replaced by a made-up bad ball (see module doc)"""
    rng = random.Random(seed); out = dict(fr); ks = sorted(k for k in fr if fr[k][0] == per); cur = None; left = 0
    for k in ks:
        p, ps, b = fr[k]
        if left <= 0:
            left = int(hold_s * FPS)
            cur = (rng.uniform(0, L), rng.uniform(0, W)) if kind == "garbage" else (rng.choice(ps)[0] if ps else None)
        left -= 1
        if kind == "garbage": nb = cur
        else:
            q = next((q for q in ps if q[0] == cur), None)
            if q is None and ps: cur = rng.choice(ps)[0]; q = next(q for q in ps if q[0] == cur)
            nb = (q[1] + 0.3, q[2]) if q else None
        out[k] = (p, ps, nb)
    return out


def capture(fr, per, noisy, seed):
    """metricalab.ours with perd, ballm (metres), the raw state captured"""
    import metricalab as ML
    got = {}; o_car, o_ps = P.carriers, P.pipeline_state
    def car_(*a, **k): r = o_car(*a, **k); got["perd"] = a[0]; got["ballm"] = r[1]; return r
    def ps_(*a, **k): r = o_ps(*a, **k); got["state"] = r[0]; return r
    P.carriers, P.pipeline_state = car_, ps_
    try: o = ML.ours(fr, per, noisy, seed=seed)
    finally: P.carriers, P.pipeline_state = o_car, o_ps
    return got["perd"], got["ballm"], got["state"], o


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), balls=(False, "flicker", "streak", True, "garbage", "sticky"), seeds=(0, 1)):
    import metricalab as ML, n0_24d_ppda as PP, n0_25d_tilt_windows as TW, n0_29f_possession as PF
    rows = []
    for g in games:
        fr, ev = PP.load_game(d, g)
        nt = ML.truth(ML.load_events(f"{d}/{g}/{g}_RawEventsData.csv"), {p: [k for k in fr if fr[k][0] == p] for p in halves})
        for per in halves:
            ks = sorted(k for k in fr if fr[k][0] == per); k0, n = ks[0], len(ks)
            own = TW.truth_own(ev, fr, per)
            seen = ~PF.gap_mask([fr[k][2] is not None for k in ks], FPS, 2); tr = PF.truth_share(own, k0, n, seen)
            true_idx = [(i, fr[k][2]) for i, k in enumerate(ks) if fr[k][2] is not None]
            tpass = sum(nt[per]["passes"].values())
            for b in balls:
                for sd in (seeds if b else (0,)):
                    if b in ("garbage", "sticky"): perd, ballm, st, o = capture(bad_ball(fr, per, b, sd), per, False, sd)
                    else: perd, ballm, st, o = capture(fr, per, b, sd)
                    nf = near_frac(perd, ballm)
                    nf_true = near_frac(perd, {i: np.array(t) for i, t in true_idx})            # TRUE ball, the same visible players
                    acc40, acc_all = accuracy(ballm, None, true_idx, sd)
                    fp = len(ballm) / max(1, n)
                    pos_ok, ev_ok = gate(acc40, nf, fp); pos_fb, _ = gate(None, nf, fp)
                    ours_p = PF.share(st, seen); agree = frame_agree(st, own, k0, seen); opass = sum(o["passes"].values())
                    name = {False: "perfect", True: "veo"}.get(b, b)
                    rows.append({"game": g, "half": per, "ball": name, "seed": sd, "near_frac": round(nf, 3), "near_frac_true_ball": round(nf_true, 3),
                                 "frames_pct": round(fp, 3), "acc40": round(acc40, 3), "acc_all": round(acc_all, 3),
                                 "poss_truth": tr, "poss_ours": ours_p, "poss_err": round(ours_p - tr, 1) if ours_p is not None and tr is not None else None,
                                 "poss_frame_agree": agree, "passes_truth": tpass, "passes_ours": opass, "passes_ratio": round(opass / max(1, tpass), 2),
                                 "possession_ok": pos_ok, "events_ok": ev_ok, "possession_ok_no_check": pos_fb})
                    print(g, per, name, sd, {k: rows[-1][k] for k in ("near_frac", "poss_frame_agree", "near_frac_true_ball", "acc40", "poss_err", "passes_ratio", "possession_ok", "events_ok", "possession_ok_no_check")}, flush=True)
    return rows


def summary(rows):
    out = {}
    for b in ("perfect", "flicker", "streak", "veo", "garbage", "sticky"):
        R = [r for r in rows if r["ball"] == b]
        if not R: continue
        f = lambda k: [r[k] for r in R if r[k] is not None]
        out[b] = {"runs": len(R), "near_frac": [round(min(f("near_frac")), 2), round(max(f("near_frac")), 2)],
                  "near_frac_true_ball": [round(min(f("near_frac_true_ball")), 2), round(max(f("near_frac_true_ball")), 2)],
                  "acc40": [round(min(f("acc40")), 2), round(max(f("acc40")), 2)], "poss_abs_err_mean": round(float(np.mean(np.abs(f("poss_err")))), 1),
                  "poss_abs_err_worst": round(float(np.max(np.abs(f("poss_err")))), 1), "poss_frame_agree": [min(f("poss_frame_agree")), max(f("poss_frame_agree"))], "passes_ratio": [min(f("passes_ratio")), max(f("passes_ratio"))],
                  "possession_ok": f"{sum(r['possession_ok'] for r in R)}/{len(R)}", "events_ok": f"{sum(r['events_ok'] for r in R)}/{len(R)}",
                  "possession_ok_no_check": f"{sum(r['possession_ok_no_check'] for r in R)}/{len(R)}"}
    return out


def dry():
    """stand-ins: the near rule, the gate and the bad balls behave"""
    perd = {0: [[1, "A", np.array([0.0, 0.0])]], 1: [[1, "A", np.array([0.0, 0.0])]], 2: []}
    assert near_frac(perd, {0: np.array([3.0, 0.0]), 1: np.array([5.0, 0.0]), 2: np.array([0.0, 0.0])}) == 1 / 3
    assert gate(0.82, 0.47, 1.0) == (True, False) and gate(None, 0.62, 1.0) == (False, False) and gate(None, 0.9, 0.5) == (True, False)
    assert gate(0.7, 0.9, 1.0) == (True, True) and gate(0.5, 0.95, 1.0) == (False, False)
    fr = {k: (1, [(101, 10.0, 10.0), (201, 50.0, 30.0)], (10.0, 10.0)) for k in range(200)}
    s = bad_ball(fr, 1, "sticky", 0)
    assert all(s[k][2] is not None and min(abs(s[k][2][0] - x) for _, x, _ in s[k][1]) < 0.5 for k in s)
    g = bad_ball(fr, 1, "garbage", 0); assert len({g[k][2] for k in g}) == 4   # 200 frames / 50 per spot
    a, full = accuracy({0: (0.0, 0.0), 1: (5.0, 0.0)}, None, [(0, (0.5, 0.0)), (1, (0.0, 0.0)), (2, (0.0, 0.0))], 0)
    assert abs(a - 1 / 3) < 1e-9 and abs(full - 1 / 3) < 1e-9
    assert frame_agree([0, 1, 2, 0], {10: "A", 11: "A", 12: "B", 13: "B"}, 10, [True] * 4) == round(1 / 3, 3)
    print("dry ok")


if __name__ == "__main__":
    if "--dry" in sys.argv: dry(); sys.exit()
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "results/qa/n0-31a/metrica_ballgrade.json"
    if "--game" in sys.argv:
        g = sys.argv[sys.argv.index("--game") + 1]; rows = main(sys.argv[1], games=(g,))
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True); json.dump({"rows": rows}, open(out, "w"), indent=1); sys.exit()
    if "--merge" in sys.argv:
        rows = [r for f in sys.argv[sys.argv.index("--merge") + 1:] if f.endswith(".json") and f != out for r in json.load(open(f))["rows"]]
    else: rows = main(sys.argv[1])
    s = summary(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"rows": rows, "summary": s}, open(out, "w"), indent=1)
    print(json.dumps(s, indent=1))
