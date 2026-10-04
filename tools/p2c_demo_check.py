"""P2c (4 Oct): the classifier self-check must not change the demo matches. Kit model fitted on each match's fit frames
(results/qa/f1c/frames/<match>/fit, as tools/f1clab.py / tools/vall_kits.py), predictions on its test frames with the
self-check on (now) and off (before, IPANEMA_CLS_MAX_FLIP=0). Free, local: python tools/p2c_demo_check.py"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K
from tools.f1clab import load
out = {}
for M in sorted(os.listdir("results/qa/f1c/frames")):
    d = f"results/qa/f1c/frames/{M}"
    if not os.path.exists(f"{d}/fit/boxes.json"): continue
    fit, test = load(f"{d}/fit"), load(f"{d}/test"); P = {}; logs = {}
    for name, v in (("before", "0"), ("now", None)):
        os.environ.pop("IPANEMA_CLS_MAX_FLIP", None)
        if v: os.environ["IPANEMA_CLS_MAX_FLIP"] = v
        L = []; m = K.KitTeamModel().fit_frames([(f, b) for _, f, b in fit], log=L.append)
        P[name] = [x for _, f, b in test for x in (m.predict_batch(f, b) if len(b) else [])]; logs[name] = [l for l in L if "classifier" in l]
    os.environ.pop("IPANEMA_CLS_MAX_FLIP", None)
    out[M] = {"people": len(P["now"]), "people that change": sum(a != b for a, b in zip(P["before"], P["now"])), "log now": logs["now"]}
    print(M, json.dumps(out[M]), flush=True)
json.dump(out, open("results/qa/p2c/demo_check.json", "w"), indent=1, ensure_ascii=False)
