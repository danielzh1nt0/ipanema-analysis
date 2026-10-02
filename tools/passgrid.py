import sys, os, json, pickle, numpy as np
sys.path.insert(0, "/home/claude/ipanema-analysis")
exec(open("tools/passlab.py").read().split("grade(base, \"current rule\")")[0])
grade(base, "current")
for kw in ({"mate_sandwich_s": 0.3}, {"mate_sandwich_s": 0.5}, {"team_sandwich_s": 0.6}, {"min_touch_s": 1.0}, {"floor_s": 0.2}, {"floor_s": 0.3}, {"min_pass_m": 8.0},
           {"state": state}, {"mate_sandwich_s": 0.3, "team_sandwich_s": 0.6}, {"floor_s": 0.2, "mate_sandwich_s": 0.3}, {"pass_by_mps": 3.0, "ballm": ballm}, {"pass_by_mps": 3.0, "ballm": ballm, "mate_sandwich_s": 0.3}):
    grade(run(**kw), ", ".join(f"{k}={'yes' if k in ('state','ballm') else v}" for k, v in kw.items()))
