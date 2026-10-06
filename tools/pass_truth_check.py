"""6 Oct: our exported passes vs Daniel's taps (Pass Tapper, results/app/passtap/db/taps/*.json).
A tap and one of our passes match when the same team and within TOL s. Prints, per clip and filter: ours, taps, matched,
found (recall) and real (precision).   python tools/pass_truth_check.py"""
import json, glob, os
TOL = 1.5
def match(ours, taps):
    used, m = set(), 0
    for t in taps:
        c = [(abs(o[0] - t[0]), i) for i, o in enumerate(ours) if i not in used and o[1] == t[1] and abs(o[0] - t[0]) <= TOL]
        if c: used.add(min(c)[1]); m += 1
    return m
def f_none(P): return P
def f_gap(g):
    def f(P):
        out = []
        for p in sorted(P, key=lambda p: p["t"]):
            if out and p["t"] - out[-1]["t"] < g: continue
            out.append(p)
        return out
    return f
FILTERS = {"as now": f_none, "drop < 1 s after previous": f_gap(1.0), "drop < 1.5 s": f_gap(1.5), "drop < 2 s": f_gap(2.0)}
for fn in sorted(glob.glob("results/app/passtap/db/taps/*.json")):
    d = json.load(open(fn)); d = d.get("data", d)
    if not d.get("done"): continue
    m, t0 = d["clip"].rsplit("_", 1); t0 = float(t0); taps = [(x["s"], x["team"]) for x in d["taps"]]
    end = 120.0
    P = [p for p in json.load(open(f"results/volume/runs/matches/{m}/stats.json"))["passes"] if t0 <= p["t"] < t0 + end]
    print(f"== {d['clip']}: {len(taps)} real passes (SFK {sum(t[1]=='A' for t in taps)}, opp {sum(t[1]=='B' for t in taps)})")
    for name, f in FILTERS.items():
        ours = [(p["t"] - t0, p["team"]) for p in f(P)]; k = match(ours, taps)
        print(f"  {name:28s} ours {len(ours):3d}  matched {k:3d}  found {k/len(taps):4.0%}  real {k/max(1,len(ours)):4.0%}")
