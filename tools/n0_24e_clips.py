"""N0-24e (24 Oct, local, free): the restart rule old vs the unseen-ball-gap rule (P.GAP_DEAD_S) on the two app clips'
exact inputs (picker_inputs.pkl, as tools/n0_20e_restarts.py) -> results/qa/n0-24e/clips.json, and the moments only one
rule has, for blind picture sheets (moments_<clip>.json; which rule had it -> moments_key.json).
    PYTHONPATH=. python tools/n0_24e_clips.py [gap_s edge_m side]"""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import ball as BL, possession as P
from n0_20e_restarts import CLIPS as _C, to_px
import glob
CLIPS = {**_C, **{os.path.basename(os.path.dirname(f)) + '@' + f.split('@')[1]: f for f in sorted(glob.glob('results/volume/cache/*/picker_inputs.pkl@*'))}}

OUT = "results/qa/n0-24e"; q = lambda *a: None


def main(gap=(2.0, 3.0, "either")):
    os.makedirs(OUT, exist_ok=True); rep, key = {}, {}
    for clip, path in CLIPS.items():
        I = pickle.load(open(path, "rb")); fps, H, L, W = I["fps"], I["H"], I["L"], I["W"]
        per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in I["per"].items()}
        ball = BL.bridge(BL.pick_v2(I["cands"], H, L, W, per=per, fps=fps, log=q), fps)
        frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0); state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q)
        n = len(per); seen = len([k for k in ballm if 0 <= k < n]) / n
        old = P.restarts(dstate, ballm, fps, L, W, gap_s=0, per=per)
        P.GAP_EDGE_M, P.GAP_SIDE = gap[1], gap[2]
        new = P.restarts(dstate, ballm, fps, L, W, gap_s=gap[0], per=per)
        gd = P.gap_dead(dstate, ballm, n, fps, L, W, gap[0], gap[1], gap[2])
        gaps_s = round(float(((gd == 3) & (np.asarray(dstate) != 3)).sum()) / fps, 1)
        mo = lambda a, b: [r for r in a if not any(abs(r["t"] - s["t"]) <= 5 for s in b)]
        only_new, only_old = mo(new, old), mo(old, new)
        both = [(r, next(s for s in new if abs(r["t"] - s["t"]) <= 5)) for r in old if any(abs(r["t"] - s["t"]) <= 5 for s in new)]
        rep[clip] = {"minutes": round(n / fps / 60, 1), "ball_seen_frac": round(seen, 2), "extra_dead_s": gaps_s, "old": old, "new": new,
                     "only_new": only_new, "only_old": only_old, "changed_in_both": [[a, b] for a, b in both if (a["kind"], a["team"]) != (b["kind"], b["team"])]}
        print(f"{clip}: {n / fps / 60:.1f} min, ball seen {seen:.0%}; restarts old {len(old)} -> new {len(new)}; only new {len(only_new)}, only old {len(only_old)}; extra dead {gaps_s} s")
        for r in new: print("   new", r)
        for r in old: print("   old", r)
        short = clip.split("-vs-")[-1][:3].lower() if "-vs-" in clip else "sfk"; short += clip.split("@")[1].split("-")[0] if "@" in clip else ""; ms = []
        cand = [("new", r) for r in only_new] + [("old", r) for r in only_old] + [("both_new", b) for a, b in both]
        for i, (who, r) in enumerate(sorted(cand, key=lambda c: c[1]["t"])):
            mid = f"{short}{i:02d}"; k_end = int(round(r["t"] * fps)); tiles = []
            for dt in np.arange(-4.0, 3.01, 0.5):
                k = int(round(k_end + dt * fps))
                if 0 <= k < n:
                    bp = ball.get(k) if isinstance(ball, dict) else None
                    tiles.append({"frame": k, "t": round(k / fps, 1), "dt": round(float(dt), 1), "ball_px": [float(bp[0]), float(bp[1])] if bp is not None else None,
                                  "spot_px": [None if r["x_m"] is None else to_px(H[k], (r["x_m"], r["y_m"]))]})
            ms.append({"id": mid, "clip": clip, "t_end": r["t"], "tiles": tiles}); key[mid] = {"rule": who, "restart": r}
        json.dump({"clips": {clip: f"{clip}/video.mp4"}, "moments": ms}, open(f"{OUT}/moments_{short}.json", "w"), indent=1)
    json.dump(rep, open(f"{OUT}/clips.json", "w"), indent=1); json.dump(key, open(f"{OUT}/moments_key.json", "w"), indent=1)
    return rep


if __name__ == "__main__":
    a = sys.argv[1:]
    main((float(a[0]), float(a[1]), a[2]) if len(a) >= 3 else (2.0, 3.0, "either"))
