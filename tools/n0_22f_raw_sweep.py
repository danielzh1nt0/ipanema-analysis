import sys, json, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "tools")
import metricalab as ML, n0_22f_regain as R
from ipanema import possession as P
"""N0-22f: regain-within-5-s from the RAW possession state (spans shorter than m s dropped), loser held >= b s; Metrica clean/streak/flicker.
    PYTHONPATH=. python tools/n0_22f_raw_sweep.py <metrica data dir> <out.json>"""
d = sys.argv[1]; res = {}
def spans_of(st, fps, join_s):
    spans = []; cur = None
    for k, s in enumerate(st):
        if s is None or s > 1: continue
        if cur and cur[0] == s and k - cur[2] <= int(join_s * fps): cur[2] = k; continue
        if cur: spans.append(cur)
        cur = [int(s), k, k]
    if cur: spans.append(cur)
    return spans
def regains(st, fps, before_s, min_span_s, join_s=0.5, win_s=5.0):
    sp = [s for s in spans_of(st, fps, join_s) if s[2] - s[1] >= min_span_s * fps]
    # merge same-team neighbours after dropping short spans
    m = []
    for s in sp:
        if m and m[-1][0] == s[0]: m[-1][2] = s[2]
        else: m.append(list(s))
    out = []
    for i in range(len(m) - 1):
        a, b = m[i], m[i + 1]
        if a[2] - a[1] < before_s * fps: continue
        back = m[i + 2] if i + 2 < len(m) else None
        out.append(back is not None and back[1] - a[2] <= win_s * fps)
    return out
orig = P.spell_state
for g in ("Sample_Game_1", "Sample_Game_2"):
    gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
    fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}; ev = ML.load_events(f"{gd}/{g}_RawEventsData.csv")
    for per in (1, 2):
        T = R.truth_regains(ev, per, ML.FPS); key = f"{g[-1]}.{per}"
        res[key] = {"truth": [len(T), R.rate(t["regained"] for t in T)]}
        for noisy in (False, "streak", "flicker"):
            got = {}
            def cap(st, *a, **k): got["raw"] = list(st); return orig(st, *a, **k)
            P.spell_state = cap
            try: ML.ours(fr, per, noisy)
            finally: P.spell_state = orig
            for ms in (0.3, 0.5, 1.0):
                for b in (1.0, 2.0):
                    x = regains(got["raw"], ML.FPS, b, ms); res[key][f"{noisy}_m{ms}_b{b}"] = [len(x), R.rate(x)]
        print(key, json.dumps(res[key]), flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
