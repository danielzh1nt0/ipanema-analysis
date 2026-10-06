"""6 Oct (P-PASS): rebuild passes offline from a time slice of picker_inputs.pkl (fetched with [fetch:...@t0-t1]) exactly as
run.analyse does, and score them against Daniel's taps (Pass Tapper). Free, local. Settings to try are passed as kwargs
to analytics.passes, plus an optional post-filter.
    python tools/passlab_taps.py"""
import sys, os, json, glob, pickle, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, possession as P, analytics as AN
q = lambda *a: None; TOL = 1.5

def load(path):
    D = pickle.load(open(path, "rb")); fps = D["fps"]; H = D["H"]; L, W = D["L"], D["W"]; off = D["offset_frames"] / fps
    n = max(D["per"]) + 1
    per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in D["per"].get(k, [])] for k in range(n)}
    cands = {k: D["cands"].get(k, []) for k in range(n)}; H = {k: H[k] for k in range(n) if k in H}
    for k in range(n):
        if k not in H: H[k] = H[max(j for j in H if j < k)] if any(j < k for j in H) else H[min(H)]
    ball = BL.bridge(BL.pick_v2(cands, H, L, W, per=per, fps=fps, log=q), fps); frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0)
    state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q); ar = {"A": True, "B": False}
    cstate = P.spell_state(state, fps, P.SPELL_TAKE_S, P.SPELL_JOIN_S)
    tvs = P.turnovers(per, frames_, cstate, ballm, fps, ar, 2.0, 5.0, min_before_s=pinfo["turnover_s"], min_after_s=pinfo["turnover_s"])
    ln = AN.lanes(per, frames_, ar, 1.5, 45.0)
    return dict(per=per, frames_=frames_, tvs=tvs, ln=ln, ar=ar, fps=fps, cstate=cstate, ballm=ballm, off=off)

def run(c, **kw):
    ps, _ = AN.passes(c["per"], c["frames_"], c["tvs"], c["ln"], c["ar"], c["fps"], state=c["cstate"], **kw)
    return [(c["off"] + p["t"], p["team"], p) for p in ps]

def score(ours, taps, t0, t1):
    ours = [o for o in ours if t0 <= o[0] < t1]; used = set(); m = 0
    for ts, tm in taps:
        cc = [(abs(o[0] - ts), i) for i, o in enumerate(ours) if i not in used and o[1] == tm and abs(o[0] - ts) <= TOL]
        if cc: used.add(min(cc)[1]); m += 1
    return len(ours), m

if __name__ == "__main__":
    for fn in glob.glob("results/app/passtap/db/taps/*.json"):
        d = json.load(open(fn)); d = d.get("data", d); m, t0 = d["clip"].rsplit("_", 1); t0 = float(t0)
        sl = glob.glob(f"results/volume/cache/{m}/picker_inputs.pkl@*")
        sl = [s for s in sl if float(s.split("@")[1].split("-")[0]) <= t0 and float(s.split("-")[-1]) >= t0 + 120]
        if not sl: print("no slice for", d["clip"]); continue
        c = load(sl[0]); taps = [(t0 + x["s"], x["team"]) for x in d["taps"]]
        n, k = score(run(c), taps, t0, t0 + 120); print(d["clip"], "taps", len(taps), "| default: ours", n, "matched", k)
