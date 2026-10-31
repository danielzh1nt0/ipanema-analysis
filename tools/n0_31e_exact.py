"""N0-31e: final-third entries / field tilt on the two app clips' exact inputs (results/volume/cache/<clip>/picker_inputs.pkl,
steps like ipanema/run.py, as tools/n0_24c_clips.py), old rule vs off-pitch balls skipped. -> results/qa/n0-31e/exact.json

    PYTHONPATH=. python tools/n0_31e_exact.py"""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, possession as P, metrics as M
from ipanema.run import periods_direction

CLIPS = {"SFKBP1109_s1200": "results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl",
         "p15u-vs-aik-2026-09-21-bd09_s2520": "results/volume/cache/p15u-vs-aik-2026-09-21-bd09_s2520/picker_inputs.pkl"}
q = lambda *a: None


def main(out="results/qa/n0-31e/exact.json"):
    rep = {}
    for clip, path in CLIPS.items():
        I = pickle.load(open(path, "rb")); fps, H, L, W = I["fps"], I["H"], I["L"], I["W"]
        per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in I["per"].items()}
        ball = BL.bridge(BL.pick_v2(I["cands"], H, L, W, per=per, fps=fps, log=q), fps)
        frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0); state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q)
        ar, conf = P.direction(state, ballm, log=q); rd = periods_direction(".", clip)
        if rd is not None: ar = {"A": bool(rd), "B": not rd}
        elif min(conf.values()) < 0.15: ar = P.direction_from_keepers(per, L, log=q) or P.direction_fallback(per, L, log=q)
        kw = dict(hold_s=M.ENTRY_HOLD_S)
        old = M.field_tilt_and_entries(state, ballm, bspeed, fps, L, ar, *M.ENTRY_REARM, onpitch=False, **kw)
        new = M.field_tilt_and_entries(state, ballm, bspeed, fps, L, ar, *M.ENTRY_REARM, W=W, onpitch=True, **kw)
        off = sum(1 for b in ballm.values() if b is not None and not (-2 <= b[0] <= L + 2 and -2 <= b[1] <= W + 2))
        r = {"fps": fps, "attack_right": ar, "ball_frames": len(ballm), "ball_off_pitch_share": round(off / max(1, len(ballm)), 3)}
        for tm in "AB":
            o = old[tm]["final_third_entries"]; n = new[tm]["final_third_entries"]; nt = {e["t"] for e in n}; ot = {e["t"] for e in o}
            r[tm] = {"old_entries": len(o), "new_entries": len(n), "old_tilt": old[tm]["field_tilt_pct"], "new_tilt": new[tm]["field_tilt_pct"],
                     "only_old": [e for e in o if e["t"] not in nt], "only_new": [e for e in n if e["t"] not in ot],
                     "old_y_off_pitch": sum(1 for e in o if not (-2 <= e["y_m"] <= W + 2))}
        rep[clip] = r; print(clip, json.dumps(r))
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(rep, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
