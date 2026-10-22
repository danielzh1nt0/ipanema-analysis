import sys, json, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "tools")
import metricalab as ML, n0_22f_regain as R
from ipanema import possession as P
"""N0-22f: regain-within-5-s straight from the counted (spell) state, loser held >= b s; Metrica clean/streak/flicker.
    PYTHONPATH=. python tools/n0_22f_spell_sweep.py <metrica data dir> <out.json>"""
d = sys.argv[1]; res = {}
def changes(cs, fps, before_s, win_s=5.0, after_min_s=0.0):
    st = [int(x) if x < 2 else None for x in cs]; n = len(st); out = []; last = None; lastk = None; run = 0
    # spans of control
    spans = []; cur = None
    for k, s in enumerate(st):
        if s is None: continue
        if cur and cur[0] == s and k - cur[2] <= int(3 * fps): cur[2] = k; continue
        if cur: spans.append(cur)
        cur = [s, k, k]
    if cur: spans.append(cur)
    for a, b in zip(spans, spans[1:]):
        if a[0] == b[0]: continue
        if (a[2] - a[1]) < before_s * fps: continue
        if (b[2] - b[1]) < after_min_s * fps and True: pass
        back = next((c for c in spans if c[1] > b[1] and c[0] == a[0]), None)
        out.append(back is not None and back[1] - a[2] <= win_s * fps)
    return out
for g in ("Sample_Game_1", "Sample_Game_2"):
    gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
    fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}; ev = ML.load_events(f"{gd}/{g}_RawEventsData.csv")
    for per in (1, 2):
        T = R.truth_regains(ev, per, ML.FPS); key = f"{g[-1]}.{per}"
        res[key] = {"truth": [len(T), R.rate(t["regained"] for t in T)]}
        for noisy in (False, "streak", "flicker"):
            o = R.ours_tvs(fr, per, noisy)
            for b in (0.5, 1.0, 2.0, 3.0):
                x = changes(o["cstate"], ML.FPS, b); res[key][f"{noisy}_b{b}"] = [len(x), R.rate(x)]
        print(key, json.dumps(res[key]), flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
