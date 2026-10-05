"""P3: score lab runs against the by-eye pair grades (results/qa/p3/grades.json, pairs.json).
    python tools/p3_score.py results/qa/p3/lab_pos.json ..."""
import json, sys
G = json.load(open("results/qa/p3/grades.json")); PR = json.load(open("results/qa/p3/pairs/pairs.json"))
GR = {n: {(p["a"], p["b"]): G[n][p["n"]] for p in PR[n]} for n in PR}


def score(lab, filt=lambda c: True):
    tot = {"r": 0, "w": 0, "?": 0, "n": 0, "ungraded": 0}; per = {}
    for n, R in lab.items():
        cand = {(c[0], c[1]): c for c in R["cands"]}; s = {"r": 0, "w": 0, "?": 0, "n": 0, "ungraded": 0}
        for a, b, *_ in R["joins"]:
            if not filt(cand[(a, b)]): continue
            s[GR.get(n, {}).get((a, b), "ungraded")] += 1
        per[n] = s
        for k in s: tot[k] += s[k]
    return tot, per


if __name__ == "__main__":
    for p in sys.argv[1:]:
        t, per = score(json.load(open(p))); print(p, t)
