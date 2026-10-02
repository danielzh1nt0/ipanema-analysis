"""B10 (2 Oct): still-ball prior before the picker (BL.pick_still), graded like B9 on the exact app inputs: SFK-BP 34 moments
+ B4 key (285), AIK 39, and the by-eye graded SFK-BP passes (real kept / fake kept). Free, local.
    PYTHONPATH=. python tools/b10lab.py            writes results/ball/b10lab.json"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL
import tools.b9lab_lib as LB

def score(skw):
    a = LB.pick(LB.A, skw); s = LB.pick(LB.S, skw); return LB.grade(a, s)

if __name__ == "__main__":
    base = score(None); print("current", base, flush=True)
    grid = {"still_px": [10.0, 20.0], "min_s": [1.0, 1.5], "boost": [0.5, 0.8], "discount": [1.0, 0.5, 0.25], "extend": [False, True]}
    if len(sys.argv) > 1 and sys.argv[1] == "quick": grid = {"still_px": [15.0], "min_s": [1.0], "boost": [0.7], "discount": [0.5], "extend": [True]}
    res = []
    for vals in itertools.product(*grid.values()):
        kw = dict(zip(grid, vals)); r = {"kw": kw, **score(kw)}; res.append(r); print(r, flush=True)
    json.dump({"current": base, "grid": grid, "all": res}, open("results/ball/b10lab.json" if len(sys.argv) == 1 else "/dev/null", "w"), indent=1)
