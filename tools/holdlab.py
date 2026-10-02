"""S8/B10 (2 Oct): grade BL.hold_still (keep a still ball when the picker teleports) on the exact app inputs of both clips:
ball keys (AIK 39, SFK-BP 34, B4 285) must not drop; then passes (87 on SFK-BP, 28 graded: real kept / fake kept) and
the counted stats. Free, local.   PYTHONPATH=. python tools/holdlab.py"""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, possession as P, analytics as AN
q = lambda *a: None
on = lambda b, f, x, y: f in b and np.hypot(b[f][0] - x, b[f][1] - y) <= 30
def load(m):
    P_ = pickle.load(open(f"results/volume/cache/{m}/picker_inputs.pkl", "rb"))
    P_["per"] = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P_["per"].items()}
    return P_
A, S = load("p15u-vs-aik-2026-09-21-bd09_s2520"), load("SFKBP1109_s1200")
agt = {int(k): v for k, v in json.load(open("reference/p15u-vs-aik-2026-09-21-bd09_s2520/ball_gt.json")).items()}
sgt = {int(k): v for k, v in json.load(open("results/volume/reference/SFKBP1109_s1200/ball_gt.json")).items() if v}
b4 = [m for m in json.load(open("results/ball/b4/key.json"))["moments"] if m["verdict"] == "ball"]
rawA = BL.pick_v2(A["cands"], A["H"], A["L"], A["W"], per=A["per"], fps=A["fps"], log=q)
rawS = BL.pick_v2(S["cands"], S["H"], S["L"], S["W"], per=S["per"], fps=S["fps"], log=q)
PA = json.load(open("results/review/passcheck_answers.json")); fake = set(PA["fake"]); unsure = set(PA["unsure"]); graded = [i for i in range(PA["graded_first_n"]) if i not in unsure]
exp = json.load(open("results/volume/runs/matches/SFKBP1109_s1200/stats.json"))["passes"]
def grade(ps):
    keep = [any(abs(p["t"] - exp[i]["t"]) < 0.3 for p in ps) for i in graded]
    return sum(k for i, k in zip(graded, keep) if i not in fake), sum(k for i, k in zip(graded, keep) if i in fake)
def stats(P_, ball):
    per, H, L, W, fps = P_["per"], P_["H"], P_["L"], P_["W"], P_["fps"]
    ball = BL.bridge(dict(ball), fps); frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0)
    state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q); ar, _ = P.direction(state, ballm, log=q)
    tvs = P.turnovers(per, frames_, state, ballm, fps, ar, 2.0, 5.0, min_before_s=pinfo["turnover_s"], min_after_s=pinfo["turnover_s"])
    ln = AN.lanes(per, frames_, ar, 1.5, 30.0); ps, _ = AN.passes(per, frames_, tvs, ln, ar, fps)
    seqs = P.sequences(state, ballm, bspeed, fps, L, ar, **pinfo["seq"])
    flips = int(sum(1 for i in range(1, len(state)) if state[i] < 2 and state[i - 1] < 2 and state[i] != state[i - 1]))
    return ps, len(tvs), len(seqs), flips
rows = []
TRIALS = [("picker as today", None, None), ("hold-still", {}, None)] + [(f"clutter px {cp} share {sh}", None, {"clutter_px": cp, "clutter_share": sh}) for cp, sh in ((40, 0.6), (40, 0.4), (60, 0.4), (40, 0.3))] + [("clutter px 40 share 0.4 + ghosts", None, {"clutter_px": 40, "clutter_share": 0.4, "ghost_s": 3.0})] + [("ghosts 1 s + hold-still", {}, {"ghost_s": 1.0})]
if len(sys.argv) > 1: TRIALS = [t for t in TRIALS if any(a in t[0] for a in sys.argv[1:])] or TRIALS
for name, kw, pkw in TRIALS:
    a = rawA if pkw is None else BL.pick_v2(A["cands"], A["H"], A["L"], A["W"], per=A["per"], fps=A["fps"], log=q, **pkw)
    s = rawS if pkw is None else BL.pick_v2(S["cands"], S["H"], S["L"], S["W"], per=S["per"], fps=S["fps"], log=q, **pkw)
    if kw is not None: a = BL.hold_still(a, A["cands"], A["H"], A["fps"], log=q, **kw); s = BL.hold_still(s, S["cands"], S["H"], S["fps"], log=q, **kw)
    keys = {"aik": int(sum(on(a, f, *g) for f, g in agt.items())), "sfk34": int(sum(on(s, f, *g[:2]) for f, g in sgt.items())), "b4": int(sum(on(s, m["frame"], m["x"], m["y"]) for m in b4))}
    psS, tvS, sqS, flS = stats(S, s); psA, tvA, sqA, flA = stats(A, a); g = grade(psS)
    r = {"name": name, **keys, "sfk": {"passes": len(psS), "real_kept": g[0], "fake_kept": g[1], "turnovers": tvS, "sequences": sqS, "flips": flS}, "aik": {"passes": len(psA), "turnovers": tvA, "sequences": sqA, "flips": flA}}
    rows.append(r); print(f"{name:22s} keys aik {keys['aik']}/39 sfk {keys['sfk34']}/34 b4 {keys['b4']}/285 | SFK passes {len(psS)} (real {g[0]}/20 fake {g[1]}/8) tv {tvS} seq {sqS} flips {flS} | AIK passes {len(psA)} tv {tvA} seq {sqA} flips {flA}", flush=True)
json.dump(rows, open("results/ball/holdlab_2026-10-02.json", "w"), indent=1)
