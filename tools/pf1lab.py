"""PF1 (3 Oct): particle-filter ball picker (ipanema/pfball.py) vs the Viterbi picker (ball.pick_v2) on the exact app inputs
of both clips: AIK 39 key, SFK-BP 34 key, B4 key (285), graded passes. Free, local.
usage: PYTHONPATH=. python tools/pf1lab.py out.json '<json list of setting dicts>'"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b9lab_lib import A, S, BL, grade, q
from ipanema import pfball as PF
out = sys.argv[1]; settings = json.loads(sys.argv[2])
res = json.load(open(out)) if os.path.exists(out) else []
for kw in settings:
    t = time.time(); kw2 = {k: tuple(v) if isinstance(v, list) else v for k, v in kw.items()}
    a = BL.bridge(PF.pick_pf(A["cands"], A["H"], A["L"], A["W"], per=A["per"], fps=A["fps"], log=q, **kw2), A["fps"])
    s = BL.bridge(PF.pick_pf(S["cands"], S["H"], S["L"], S["W"], per=S["per"], fps=S["fps"], log=q, **kw2), S["fps"])
    r = {"kw": kw, **grade(a, s), "sec": round(time.time() - t)}; res.append(r); print(r, flush=True)
    json.dump(res, open(out, "w"), indent=1)
