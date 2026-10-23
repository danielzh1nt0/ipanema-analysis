"""N0-23a (23 Oct, local, free): 'forward pass within 3 s of winning it' on the two app clips' exact inputs
(results/volume/cache/<clip>/picker_inputs.pkl), steps like ipanema/run.py, old (IPANEMA_FWD_FROM_WIN=0) vs new.
Writes results/qa/n0-23a/clips.json and moments.json (the wins whose answer changed, for picture sheets).
    PYTHONPATH=. python tools/n0_23a_clips.py"""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, possession as P, analytics as AN
from ipanema.run import periods_direction

CLIPS = {"SFKBP1109_s1200": "results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl",
         "p15u-vs-aik-2026-09-21-bd09_s2520": "results/volume/cache/p15u-vs-aik-2026-09-21-bd09_s2520/picker_inputs.pkl"}
OUT = "results/qa/n0-23a"; q = lambda *a: None


def kept_turnovers(per, frames_, cstate, ballm, fps, ar, pinfo, ln):
    tvs = P.turnovers(per, frames_, cstate, ballm, fps, ar, 2.0, 5.0, min_before_s=pinfo["turnover_s"], min_after_s=pinfo["turnover_s"])
    ps, _ = AN.passes(per, frames_, tvs, ln, ar, fps, state=cstate)
    return P.confirm_losses(tvs, cstate, ps, fps, P.LOSS_KEEP_S)[0]


def summ(tvs):
    v = [t["time_to_forward_pass"] for t in tvs]; x = [t for t in v if t is not None]
    return {"wins": len(v), "fwd3_pct": round(100 * float(np.mean([t is not None and t <= 3 for t in v]))) if v else None,
            "median_s": round(float(np.median(x)), 2) if x else None, "under_0_5s": sum(t < 0.5 for t in x)}


def main():
    rep, moments = {}, []
    for clip, path in CLIPS.items():
        I = pickle.load(open(path, "rb")); fps, H, L, W = I["fps"], I["H"], I["L"], I["W"]
        per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in I["per"].items()}
        ball = BL.bridge(BL.pick_v2(I["cands"], H, L, W, per=per, fps=fps, log=q), fps)
        frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0); state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q)
        ar, conf = P.direction(state, ballm, log=q); rd = periods_direction("/nonexistent", clip)
        if rd is not None: ar = {"A": bool(rd), "B": not rd}
        elif min(conf.values()) < 0.15: ar = P.direction_from_keepers(per, L, log=q) or P.direction_fallback(per, L, log=q)
        cstate = P.spell_state(state, fps, P.SPELL_TAKE_S, P.SPELL_JOIN_S); ln = AN.lanes(per, frames_, ar, 1.5, 45.0)   # Settings lane_half, max_lane
        res = {}
        for name, on in (("old", False), ("new", True)):
            P.FWD_FROM_WIN = on; res[name] = kept_turnovers(per, frames_, cstate, ballm, fps, ar, pinfo, ln)
        P.FWD_FROM_WIN = True
        assert [t["frame"] for t in res["old"]] == [t["frame"] for t in res["new"]]
        rows = []
        for a, b in zip(res["old"], res["new"]):
            rows.append({"t_lost": a["t"], "t_won": a["t_won"], "won_by": a["won_by"], "old": a["time_to_forward_pass"], "new": b["time_to_forward_pass"]})
            ya = a["time_to_forward_pass"] is not None and a["time_to_forward_pass"] <= 3; yb = b["time_to_forward_pass"] is not None and b["time_to_forward_pass"] <= 3
            if ya != yb or (a["time_to_forward_pass"] or 9) < 0.5 or (b["time_to_forward_pass"] or 9) < 0.5:   # changed answer, or an implausibly quick pass
                k0, kl = a["frame"], a["frame_lost"]
                w0 = next((j for j in range(kl + 1, k0) if frames_[j]["team"] == a["won_by"] and frames_[j]["carrier"] is not None), k0)
                ends = [k0 + int(round((a["time_to_forward_pass"] or 0) * fps)), w0 + int(round((b["time_to_forward_pass"] or 0) * fps))]
                f_from, f_to = min(kl, w0) - int(0.5 * fps), max(ends + [k0]) + int(1.5 * fps)
                step = max(int(round(0.5 * fps)), int(np.ceil((f_to - f_from) / 15)))
                tiles = []
                for k in range(max(0, f_from), min(len(per), f_to + 1), step):
                    bp = ball.get(k); tiles.append({"frame": k, "dt": round((k - kl) / fps, 1), "ball_px": [float(bp[0]), float(bp[1])] if bp is not None else None})
                moments.append({"id": f"{'sfk' if clip.startswith('SFK') else 'aik'}{len(moments):02d}", "clip": clip, "won_by": a["won_by"], "attack_right": ar[a["won_by"]],
                                "t_lost": a["t"], "t_won": a["t_won"], "t_first_touch": round(w0 / fps, 2), "old": a["time_to_forward_pass"], "new": b["time_to_forward_pass"],
                                "tiles": tiles})
        rep[clip] = {"attack_right": ar, "old": summ(res["old"]), "new": summ(res["new"]),
                     "per_team": {tm: {k: summ([t for t in res[k] if t["won_by"] == tm]) for k in ("old", "new")} for tm in ("A", "B")}, "rows": rows}
        print(clip, json.dumps({k: rep[clip][k] for k in ("attack_right", "old", "new", "per_team")}))
        for r in rows: print("   ", r)
    os.makedirs(OUT, exist_ok=True)
    json.dump(rep, open(f"{OUT}/clips.json", "w"), indent=1); json.dump(moments, open(f"{OUT}/moments.json", "w"), indent=1)


if __name__ == "__main__":
    main()
