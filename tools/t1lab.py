"""T1 (2 Oct, local CPU): does removing / lowering the guesses the 3-frame scorer doubts help the ball picker?
Exact app inputs of both clips (results/volume/cache/<clip>/picker_inputs.pkl), scorer outputs from the free runner
(results/free/t1/<clip>.npz, tools/t1_score.py). Graded like tools/picktune.py: AIK 39-moment blind key, SFK-BP 34 moments,
B4 key (SFK-BP). Also 'top guess on the ball' at the key moments (finder alone, before the picker).
Fair use: AIK was never in any scorer's training; for SFK-BP only the sfk* models are leak-free (scorer_all saw SFK-BP clicks
inside the clip window). A setting would only be kept if AIK goes up and SFK-BP goes down on neither key (leak-free models).
    python tools/t1lab.py        -> results/ball/t1/t1lab.json"""
import sys, os, json, pickle, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, ballfilter as BF
q = lambda *a: None
on = lambda b, f, x, y: f in b and np.hypot(b[f][0] - x, b[f][1] - y) <= 30
AIK, SFK = "p15u-vs-aik-2026-09-21-bd09_s2520", "SFKBP1109_s1200"
def load(m):
    P = pickle.load(open(f"results/volume/cache/{m}/picker_inputs.pkl", "rb"))
    P["per"] = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P["per"].items()}
    z = np.load(os.environ.get("T1_DIR", "results/free/t1") + f"/{m}.npz"); P["scores"] = {k: BF.unpack(z["fr"], z[k]) for k in z["models"].tolist()}
    return P
A, S = load(AIK), load(SFK)
agt = {int(k): v for k, v in json.load(open(f"reference/{AIK}/ball_gt.json")).items()}
sgt = {int(k): v for k, v in json.load(open(f"results/volume/reference/{SFK}/ball_gt.json")).items() if v}
b4 = [m for m in json.load(open("results/ball/b4/key.json"))["moments"] if m["verdict"] == "ball"]
models = [k for k in A["scores"] if k in S["scores"]]
for P in (A, S):                                                  # average of the leak-free models = one more 'model'
    ks = [k for k in models if k.startswith("sfk")]
    P["scores"]["sfk_mean"] = {f: np.mean([P["scores"][k][f] for k in ks], 0) for f in P["scores"][ks[0]]}
models.append("sfk_mean")
def pick(P, c): return BL.bridge(BL.pick_v2(c, P["H"], P["L"], P["W"], per=P["per"], fps=P["fps"], log=q), P["fps"])
def top(c, gt):  # the single strongest guess is on the ball
    return int(sum(1 for f, g in gt.items() if c.get(f) and on({f: max(c[f], key=lambda r: r[2])[:2]}, f, g[0], g[1])))
def score(ca, cs):
    a = pick(A, ca); s = pick(S, cs)
    return {"aik": int(sum(on(a, f, *g[:2]) for f, g in agt.items())), "sfk34": int(sum(on(s, f, *g[:2]) for f, g in sgt.items())),
            "b4": int(sum(on(s, m["frame"], m["x"], m["y"]) for m in b4)),
            "aik_top": top(ca, agt), "sfk34_top": top(cs, sgt)}
t0 = time.time(); base = score(A["cands"], S["cands"]); print("app picker today", base, round(time.time() - t0), "s", flush=True)
grid = [("drop", 0.05)] if os.environ.get("T1_QUICK") else [("drop", t) for t in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5)] + [("mult", t) for t in (0.25, 0.5, 1.0)] + [("blend", t) for t in (0.25, 0.5)]
res = []
for mdl in models:
    for mode, t in grid:
        r = {"model": mdl, "mode": mode, "t": t, **score(BF.rescore(A["cands"], A["scores"][mdl], mode, t), BF.rescore(S["cands"], S["scores"][mdl], mode, t))}
        r["kept_per_frame_aik"] = round(float(np.mean([len(v) for v in BF.rescore(A["cands"], A["scores"][mdl], mode, t).values()])), 1) if mode == "drop" else None
        res.append(r); print(r, flush=True)
fair = lambda r: r["model"].startswith("sfk")
ok = [r for r in res if fair(r) and r["aik"] > base["aik"] and r["sfk34"] >= base["sfk34"] and r["b4"] >= base["b4"]]
ok.sort(key=lambda r: (-(r["aik"] + r["sfk34"]), -r["b4"]))
of = {"aik": len(agt), "sfk34": len(sgt), "b4": len(b4)}
json.dump({"base": base, "of": of, "passing": ok, "all": res}, open(os.environ.get("T1_JSON", "results/ball/t1/t1lab.json"), "w"), indent=1)
print("passing:", ok[:5])
