"""E2 (28 Sep, worker): third batch of who-has-the-ball moments on the SFK-BP clip, so possession fixes are graded on
100+ clear moments instead of 61. Frames are chosen blind (an even grid, at least 1.2 s from every earlier moment and from
each other) - NOT where the models disagree, so the key is not tilted towards any model.
Writes results/review/who_moments3.json in the same format as who_moments2.json (players + our ball pick for +-24 frames),
which tools/who_strips.py turns into picture strips on the free runner. Local, free. Run with PYTHONPATH=."""
import json, numpy as np

def pick_frames(n, taken, gap=36, step=23, edge=30):
    """even grid over the clip, keeping frames >= gap from every taken frame and from each other"""
    out = []
    for k in range(edge, n - edge, step):
        if all(abs(k - t) >= gap for t in list(taken) + out): out.append(k)
    return out

if __name__ == "__main__":
    exec(open("tools/fusedlab.py").read().split("q = lambda *a: None")[0].split('"""', 2)[2])
    import pickle
    q = lambda *a: None
    C = "results/volume/cache/SFKBP1109_s1200/"
    def load(f):
        o = pickle.load(open(C + f, "rb")); o = o[0] if isinstance(o, tuple) else o
        return {int(k): [(float(a), float(b), float(c)) for a, b, c in v] for k, v in o.items()}
    cd = BL.fuse_candidates(load("ball_cands_clicks_round_20260928_0108.pkl"), load("ball_cands_wasb_1790008894_t2x2_thr0.05.pkl"))
    ball = BL.bridge(BL.pick_v2(cd, H, L, W, per=new, fps=fps, log=q), fps)
    taken = [m["frame"] for f in ("who_moments.json", "who_moments2.json") for m in json.load(open("results/review/" + f))["moments"]]
    frames = pick_frames(n, taken)
    out = {"moments": [], "players": {}, "ball": {}}
    for k in frames:
        out["moments"].append({"t": round(k / fps, 1), "frame": k, "ours": None})
        for d_ in range(-24, 25, 12):
            f = k + d_
            out["players"][str(f)] = [[pid, t, [float(px[0]), float(px[1])], bool(fl)] for pid, t, px, fl in R.get(str(f), []) if px is not None]
            if f in ball: out["ball"][str(f)] = [round(float(ball[f][0]), 1), round(float(ball[f][1]), 1)]
    json.dump(out, open("results/review/who_moments3.json", "w"))
    print(len(frames), "new moments", frames[:10], "...")
