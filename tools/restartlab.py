"""S5 (2 Oct): restarts on the SFK-BP clip (exact app inputs) vs Veo's event list for minutes 20-25 (7 restarts: 5 throw-ins,
1 corner, 1 free kick): current picker vs the play-on spare-ball rule (B4c), and the restart-type fix. Free, local."""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, possession as P
P_ = pickle.load(open("results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl", "rb")); fps = P_["fps"]; H = P_["H"]; L, W = P_["L"], P_["W"]
per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P_["per"].items()}
q = lambda *a: None
for name, kw in (("picker as in the app", {}), ("picker + play-on spare-ball rule (B4c)", {"recur_r": 2.0, "recur_play_s": 5})):
    ball = BL.bridge(BL.pick_v2(P_["cands"], H, L, W, per=per, fps=fps, log=q, **kw), fps)
    frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0); state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q)
    for rk in ({}, {"robust": True}):
        try: rst = P.restarts(dstate, ballm, fps, L, W, **rk)
        except TypeError: continue
        kinds = {}
        for r in rst: kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        print(f"{name:42s} {'robust type' if rk else 'old type  '} -> {len(rst)} restarts {kinds}  at " + ", ".join(f"{r['t']:.0f}" for r in rst))
print("Veo minutes 20-24: throw-ins at ~20, 21, 22, 22, 23; corner 24; foul/free kick 24 (7 restarts)")
ball = BL.bridge(BL.pick_v2(P_["cands"], H, L, W, per=per, fps=fps, log=q), fps)
frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0); state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q)
for jg in (1.5, 3.0, 5.0, 8.0):
    for ms in (2.0, 3.0):
        rst = P.restarts(dstate, ballm, fps, L, W, min_s=ms, join_gap_s=jg)
        print(f"join gap {jg} s, min {ms} s -> {len(rst)} at " + ", ".join(f"{r['t']:.0f}{r['kind'][0]}" for r in rst))
