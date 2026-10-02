"""S8 (2 Oct): grade a retrained ball finder on the SFK-BP test clip exactly as the app would use it: its candidates
(results/kaggle/<run>/cands_new_SFKBP1109_s1200.pkl) fused with the clip's WASB guesses, through the picker, against the
34-moment key, the B4 key (285) and the 28 graded passes (real kept / fake kept) + pass/turnover/sequence counts.
The old finder's candidates from the same job (cands_old_...) must reproduce the app's inputs (sanity row).
    PYTHONPATH=. python tools/newfinder_grade.py results/kaggle/ballfinder_neg"""
import sys, os, json, glob, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, possession as P, analytics as AN
q = lambda *a: None; RUN = sys.argv[1] if len(sys.argv) > 1 else "results/kaggle/ballfinder_neg"; CLIP = "SFKBP1109_s1200"
on = lambda b, f, x, y: f in b and np.hypot(b[f][0] - x, b[f][1] - y) <= 30
P_ = pickle.load(open(f"results/volume/cache/{CLIP}/picker_inputs.pkl", "rb")); fps, H, L, W = P_["fps"], P_["H"], P_["L"], P_["W"]
per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P_["per"].items()}
sgt = {int(k): v[:2] for k, v in json.load(open(f"results/volume/reference/{CLIP}/ball_gt.json")).items() if v}
b4 = [m for m in json.load(open("results/ball/b4/key.json"))["moments"] if m["verdict"] == "ball"]
PA = json.load(open("results/review/passcheck_answers.json")); fake = set(PA["fake"]); unsure = set(PA["unsure"]); graded = [i for i in range(PA["graded_first_n"]) if i not in unsure]
exp = json.load(open(f"results/volume/runs/matches/{CLIP}/stats.json"))["passes"]
def load(p):
    o = pickle.load(open(p, "rb")); o = o[0] if isinstance(o, tuple) else o
    return {int(k): [(float(a), float(b), float(c)) for a, b, c in v] for k, v in o.items()}
wasbs = {os.path.basename(p): load(p) for p in glob.glob(f"results/volume/cache/{CLIP}/ball_cands_wasb_*.pkl")}
def same(a, b):
    ks = sorted(set(a) & set(b)); n = sum(1 for k in ks if len(a[k]) == len(b[k]) and all(abs(x[0] - y[0]) < 0.5 and abs(x[1] - y[1]) < 0.5 for x, y in zip(sorted(a[k]), sorted(b[k])))); return n, len(ks)
def run(cands):
    ball = BL.bridge(BL.pick_v2(cands, H, L, W, per=per, fps=fps, log=q), fps)
    keys = {"sfk34": int(sum(on(ball, f, *g) for f, g in sgt.items())), "b4": int(sum(on(ball, m["frame"], m["x"], m["y"]) for m in b4))}
    frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0); state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q); ar, _ = P.direction(state, ballm, log=q)
    tvs = P.turnovers(per, frames_, state, ballm, fps, ar, 2.0, 5.0, min_before_s=pinfo["turnover_s"], min_after_s=pinfo["turnover_s"])
    ln = AN.lanes(per, frames_, ar, 1.5, 30.0); ps, _ = AN.passes(per, frames_, tvs, ln, ar, fps); seqs = P.sequences(state, ballm, bspeed, fps, L, ar, **pinfo["seq"])
    keep = [any(abs(p["t"] - exp[i]["t"]) < 0.3 for p in ps) for i in graded]
    flips = int(sum(1 for i in range(1, len(state)) if state[i] < 2 and state[i - 1] < 2 and state[i] != state[i - 1]))
    return {**keys, "passes": len(ps), "real_kept": sum(k for i, k in zip(graded, keep) if i not in fake), "fake_kept": sum(k for i, k in zip(graded, keep) if i in fake), "turnovers": len(tvs), "sequences": len(seqs), "flips": flips}
out = {"app inputs": run(P_["cands"])}; print("app inputs (picker_inputs)", out["app inputs"], flush=True)
for tag in ("old", "new"):
    p = f"{RUN}/cands_{tag}_{CLIP}.pkl"
    if not os.path.exists(p): print(tag, "missing"); continue
    rf = load(p)
    for wn, wb in sorted(wasbs.items()):
        fused = BL.fuse_candidates(rf, wb); s = same(fused, P_["cands"])
        r = run(fused); out[f"{tag} + {wn}"] = {**r, "same_as_app": s}; print(f"{tag:3s} + {wn:48s} same as app {s[0]}/{s[1]} |", r, flush=True)
json.dump(out, open(f"{RUN}/grade.json", "w"), indent=1)
