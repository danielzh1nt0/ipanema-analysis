"""1 Oct: tune the ball picker (pick_v2) for free, offline, on the EXACT app inputs of both clips (saved by the app run as
cache/<match>/picker_inputs.pkl and fetched from Modal): AIK (39-moment blind key) and SFK-BP (34 moments + B4 key).
First version used an older offline SFK-BP setup and promised 26->28; the app gave 29->28 -> fixed here.
A setting is kept only if AIK goes up AND SFK-BP goes down on neither key. Writes results/ball/picktune.json.
The 'default' row is the setting the app ran on the day (passed explicitly so this file stays valid after defaults change)."""
import sys, os, json, pickle, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL
q = lambda *a: None
on = lambda b, f, x, y: f in b and np.hypot(b[f][0] - x, b[f][1] - y) <= 30
def load(m):
    P = pickle.load(open(f"results/volume/cache/{m}/picker_inputs.pkl", "rb"))
    P["per"] = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P["per"].items()}
    return P
def pick(P, kw): return BL.bridge(BL.pick_v2(P["cands"], P["H"], P["L"], P["W"], per=P["per"], fps=P["fps"], log=q, **kw), P["fps"])
A, S = load("p15u-vs-aik-2026-09-21-bd09_s2520"), load("SFKBP1109_s1200")
agt = {int(k): v for k, v in json.load(open("reference/p15u-vs-aik-2026-09-21-bd09_s2520/ball_gt.json")).items()}
sgt = {int(k): v for k, v in json.load(open("results/volume/reference/SFKBP1109_s1200/ball_gt.json")).items() if v}
b4 = [m for m in json.load(open("results/ball/b4/key.json"))["moments"] if m["verdict"] == "ball"]
def score(kw):
    a = pick(A, kw); s = pick(S, kw)
    return {"aik": int(sum(on(a, f, *g) for f, g in agt.items())), "sfk34": int(sum(on(s, f, *g[:2]) for f, g in sgt.items())),
            "b4": int(sum(on(s, m["frame"], m["x"], m["y"]) for m in b4))}
OLD = {"conf_w": 1.5, "near_w": 1.0, "jump_cost": 6.0, "miss_cost": 3.0}
base = score(OLD); print("app setting until 1 Oct", base, flush=True)
grid = {"conf_w": [1.5, 2.5, 4.0], "near_w": [1.0, 0.75, 0.5], "jump_cost": [6.0, 3.0], "miss_cost": [3.0, 4.0, 5.0]}
res = []
for vals in itertools.product(*grid.values()):
    kw = dict(zip(grid, vals)); r = {"kw": kw, **score(kw)}; res.append(r); print(r, flush=True)
ok = [r for r in res if r["aik"] > base["aik"] and r["sfk34"] >= base["sfk34"] and r["b4"] >= base["b4"]]
ok.sort(key=lambda r: (-(r["aik"] + r["sfk34"]), -r["b4"]))
json.dump({"old_setting": OLD, "old": base, "of": {"aik": len(agt), "sfk34": len(sgt), "b4": len(b4)}, "grid": grid, "passing": ok, "all": res},
          open("results/ball/picktune.json", "w"), indent=1)
print("best passing:", ok[:5])
