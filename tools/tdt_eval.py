"""6 Oct (P-PASS): passes from the SoccerNet team ball-action model scores (results/kaggle/tdeed_team/tdt/<clip>.npz) vs
Daniel's taps: peaks of the pass score (PASS + HIGH PASS + CROSS + FREE KICK, both sides) above THR, at least NMS s apart;
team = the side with the higher score.  python tools/tdt_eval.py"""
import json, glob, numpy as np
KICK = [0, 3, 5, 10]
def events(npz, thr=0.3, nms=0.6, kick=KICK):
    d = np.load(npz); S = d["scores"]; fps = float(d["fps"])
    L = sum(S[:, 1 + 2 * c] for c in kick); R = sum(S[:, 2 + 2 * c] for c in kick); P = L + R
    idx = [i for i in range(1, len(P) - 1) if P[i] >= thr and P[i] >= P[i - 1] and P[i] >= P[i + 1]]
    idx.sort(key=lambda i: -P[i]); keep = []
    for i in idx:
        if all(abs(i - j) / fps >= nms for j in keep): keep.append(i)
    return sorted((i / fps, "L" if L[i] >= R[i] else "R", float(P[i])) for i in keep)
def score(ev, taps, side_to_team=None, tol=1.5):
    used, m, mt = set(), 0, 0
    for t in taps:
        cc = [(abs(e[0] - t["s"]), i) for i, e in enumerate(ev) if i not in used and abs(e[0] - t["s"]) <= tol]
        if cc:
            i = min(cc)[1]; used.add(i); m += 1
            if side_to_team and side_to_team[ev[i][1]] == t["team"]: mt += 1
    return m, mt
if __name__ == "__main__":
    for fn in glob.glob("results/app/passtap/db/taps/*.json"):
        d = json.load(open(fn)); d = d.get("data", d); npz = f"results/kaggle/tdeed_team/tdt/{d['clip']}.npz"; taps = d["taps"]
        print("==", d["clip"], len(taps), "taps")
        for thr in (0.2, 0.3, 0.4, 0.5):
            ev = events(npz, thr)
            m1, a1 = score(ev, taps, {"L": "A", "R": "B"}); _, a2 = score(ev, taps, {"L": "B", "R": "A"})
            print(f"thr {thr}: model {len(ev)} matched {m1}  team right if L=SFK {a1} / if L=opp {a2}")
