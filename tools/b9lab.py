"""B9 (2 Oct): picker options against ALL keys at once, on the exact app inputs: SFK-BP 34 moments + B4 key (285), AIK 39,
and the 30 by-eye graded passes on the SFK-BP clip (8 fake, 20 real). Free, local. Writes results/ball/b9lab.json."""
import sys, os, json, pickle, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, possession as P, analytics as AN
q = lambda *a: None
on = lambda b, f, x, y: f in b and np.hypot(b[f][0] - x, b[f][1] - y) <= 30
def load(m):
    Pp = pickle.load(open(f"results/volume/cache/{m}/picker_inputs.pkl", "rb"))
    Pp["per"] = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in Pp["per"].items()}
    return Pp
A, S = load("p15u-vs-aik-2026-09-21-bd09_s2520"), load("SFKBP1109_s1200")
agt = {int(k): v for k, v in json.load(open("reference/p15u-vs-aik-2026-09-21-bd09_s2520/ball_gt.json")).items()}
sgt = {int(k): v for k, v in json.load(open("results/volume/reference/SFKBP1109_s1200/ball_gt.json")).items() if v}
b4 = [m for m in json.load(open("results/ball/b4/key.json"))["moments"] if m["verdict"] == "ball"]
exp = json.load(open("results/volume/runs/matches/SFKBP1109_s1200/stats.json"))["passes"]
G = json.load(open("results/review/passcheck_answers.json")); fake = set(G["fake"]); unsure = set(G["unsure"]); graded = [i for i in range(G["graded_first_n"]) if i not in unsure]
def pick(Pp, kw): return BL.bridge(BL.pick_v2(Pp["cands"], Pp["H"], Pp["L"], Pp["W"], per=Pp["per"], fps=Pp["fps"], log=q, **kw), Pp["fps"])
def passes_of(ball):
    fps, H, L, W, per = S["fps"], S["H"], S["L"], S["W"], S["per"]
    frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0); state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q)
    ar, _ = P.direction(state, ballm, log=q); tvs = P.turnovers(per, frames_, state, ballm, fps, ar, 2.0, 5.0, min_before_s=pinfo["turnover_s"], min_after_s=pinfo["turnover_s"])
    ln = AN.lanes(per, frames_, ar, 1.5, 30.0); ps, _ = AN.passes(per, frames_, tvs, ln, ar, fps); return ps
def score(kw):
    a = pick(A, kw); s = pick(S, kw); ps = passes_of(s)
    keep = [any(abs(p["t"] - exp[i]["t"]) < 0.3 for p in ps) for i in graded]
    return {"aik": int(sum(on(a, f, *g) for f, g in agt.items())), "sfk34": int(sum(on(s, f, *g[:2]) for f, g in sgt.items())), "b4": int(sum(on(s, m["frame"], m["x"], m["y"]) for m in b4)),
            "passes": len(ps), "real_kept": int(sum(k for i, k in zip(graded, keep) if i not in fake)), "fake_kept": int(sum(k for i, k in zip(graded, keep) if i in fake))}
base = score({}); print("current", base, flush=True)
grid = {"off_cost": [0.0, 0.3, 1.0], "px_w": [0.02, 0.05, 0.1], "jump_cost": [6.0, 12.0]}
res = []
for vals in itertools.product(*grid.values()):
    kw = dict(zip(grid, vals)); r = {"kw": kw, **score(kw)}; res.append(r); print(r, flush=True)
json.dump({"current": base, "grid": grid, "all": res}, open("results/ball/b9lab.json", "w"), indent=1)
