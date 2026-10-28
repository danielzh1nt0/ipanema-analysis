"""N0-28b (28 Oct, local, free): the app clips' sequences on the exact inputs (results/volume/cache/<clip>/picker_inputs.pkl,
steps like ipanema/run.py via tools/n0_25e_clips.clip_inputs): old vs split at the restart step's stoppages vs split +
gain from the first / last 0.5 s. Count, duration spread, gain missing, and which sequences change.
    PYTHONPATH=. python tools/n0_28b_clips.py"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P
from n0_25e_clips import CLIPS, clip_inputs

VARIANTS = {"old": dict(split_stop=False, gain="exact"), "split": dict(split_stop=True, gain="exact"), "split_edge": dict(split_stop=True, gain="edge0.5")}


def main():
    rep = {}
    for clip, path in CLIPS.items():
        D = clip_inputs(path, clip); a = (D["cstate"], D["ballm"], D["bspeed"], D["fps"], D["L"], D["ar"])
        sp = P.stoppage_spans(D["dstate"], D["ballm"], D["fps"], D["L"], D["W"]); r = {"stoppages": len(sp)}; runs = {}
        for v, kw in VARIANTS.items():
            S = P.sequences(*a, **D["pinfo"]["seq"], stops=sp, **kw); runs[v] = S; d = [s["duration_s"] for s in S]
            r[v] = {"count": len(S), "dur_q25_50_75_90": [round(float(np.percentile(d, p)), 1) for p in (25, 50, 75, 90)], "dur_max": max(d),
                    "gain_missing": sum(1 for s in S if s["gain_m"] is None), "gain_median": float(np.median([s["gain_m"] for s in S if s["gain_m"] is not None]))}
        key = lambda s: (s["team"], s["start"], s["end"])
        old = {key(s) for s in runs["old"]}; new = {key(s) for s in runs["split"]}
        r["cut"] = [{"old": [s["team"], s["t_start"], s["t_end"], s["duration_s"]], "into": [[q["t_start"], q["t_end"]] for q in runs["split"] if q["team"] == s["team"] and q["start"] >= s["start"] and q["end"] <= s["end"]]}
                    for s in runs["old"] if key(s) not in new]
        r["gain_filled"] = [[s["team"], s["t_start"], s["t_end"], q["gain_m"]] for s, q in zip(runs["split"], runs["split_edge"]) if s["gain_m"] is None and q["gain_m"] is not None]
        r["stop_spans_s"] = [[round(x / D["fps"], 1), round(y / D["fps"], 1)] for x, y in sp]
        rep[clip] = r; print(clip, json.dumps(r)[:1500])
    os.makedirs("results/qa/n0-28b", exist_ok=True); json.dump(rep, open("results/qa/n0-28b/clips.json", "w"), indent=1)
    if "--write-moments" in sys.argv: write_moments(rep)


def write_moments(rep, seed=28, controls=4):
    """blind sheets: every stoppage span the split cuts at + `controls` windows of the same length with no stoppage,
    shuffled under neutral ids; which is which goes to key_SEALED.json"""
    import random
    rnd = random.Random(seed); ms = []; key = {}
    for clip, r in rep.items():
        sp = r["stop_spans_s"]; dur = 300.0
        for a, b in sp: ms.append({"clip": clip, "a": a, "b": b, "kind": "stoppage"})
        for _ in range(controls):
            for _try in range(200):
                ln = rnd.choice([b - a for a, b in sp]); a = round(rnd.uniform(2, dur - ln - 2), 1); b = round(a + ln, 1)
                if all(b + 3 < x or a - 3 > y for x, y in sp): ms.append({"clip": clip, "a": a, "b": b, "kind": "control"}); break
    rnd.shuffle(ms); out = []
    for i, m in enumerate(ms):
        mid = f"s{i:02d}"; key[mid] = m["kind"]
        ts = [m["a"] - 1.0] + [round(m["a"] + (m["b"] - m["a"]) * j / 5, 2) for j in range(6)] + [m["b"] + 1.0]
        out.append({"id": mid, "clip": m["clip"], "a": m["a"], "b": m["b"], "times": [round(max(0.0, t), 2) for t in ts]})
    json.dump(out, open("results/qa/n0-28b/moments.json", "w"), indent=1); json.dump(key, open("results/qa/n0-28b/key_SEALED.json", "w"), indent=1)
    print(len(out), "moments ->", "results/qa/n0-28b/moments.json")


if __name__ == "__main__":
    main()
