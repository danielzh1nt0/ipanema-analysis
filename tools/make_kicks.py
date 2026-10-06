"""6 Oct (P-PASS): overrides/<match>_kicks.json from the full-match ball-action model output (results/kaggle/tdeed_full/tdf/
<match>_<t0>.json, frames at 25 fps from t0): pass moments = PASS, HIGH PASS, CROSS, FREE KICK with score >= THR, merged
when < 0.8 s apart, in video seconds.   python tools/make_kicks.py [thr]"""
import sys, json, glob, collections
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2; KICK = {"PASS", "HIGH PASS", "CROSS", "FREE KICK"}
by = collections.defaultdict(list)
for fn in glob.glob("results/kaggle/tdeed_full/tdf/*_*.json"):
    d = json.load(open(fn))
    if "match" not in d: continue
    by[d["match"]] += [d["t0"] + e["frame"] / d["fps"] for e in d["predictions"] if e["label"] in KICK and e["confidence"] >= THR]
for m, P in by.items():
    Q = []
    for t in sorted(P):
        if not Q or t - Q[-1] > 0.8: Q.append(round(t, 2))
    json.dump({"match": m, "source": "T-DEED SoccerNetBall_challenge1 (kaggle/tdeed_full.py)", "threshold": THR, "kicks": Q}, open(f"overrides/{m}_kicks.json", "w"))
    print(m, len(Q), "pass moments")
