"""S3b (3 Oct): fair grade of the pasted-ball finder (results/kaggle/ballfinder_paste) vs the old finder from the SAME job
on the SFK-BP test clip: (1) finder alone, top guess on the 34-key and the B4 key; (2) through the picker with a small
re-tune of the weights for each finder (conf_w x near_w x WASB weight), so the new finder is not judged on the old one's
tuning. Writes results/ball/s3b/grade.json.
    PYTHONPATH=. python tools/s3b_grade.py [results/kaggle/ballfinder_paste]"""
import sys, os, json, glob, pickle, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL
RUN = sys.argv[1] if len(sys.argv) > 1 else "results/kaggle/ballfinder_paste"; CLIP = "SFKBP1109_s1200"
def load(p):
    o = pickle.load(open(p, "rb")); o = o[0] if isinstance(o, tuple) else o
    return {int(k): [(float(a), float(b), float(c)) for a, b, c in v] for k, v in o.items()}
def top_hits(c, key):
    n = 0
    for f, x, y in key:
        g = c.get(f, [])
        if g:
            b = max(g, key=lambda t: t[2]); n += np.hypot(b[0] - x, b[1] - y) <= 30
    return int(n)
def picker_hits(ball, sgt, b4):
    on = lambda f, x, y: f in ball and np.hypot(ball[f][0] - x, ball[f][1] - y) <= 30
    return int(sum(on(f, *g) for f, g in sgt.items())), int(sum(on(*m) for m in b4))
def main(grid=None):
    P_ = pickle.load(open(f"results/volume/cache/{CLIP}/picker_inputs.pkl", "rb")); fps, H, L, W = P_["fps"], P_["H"], P_["L"], P_["W"]
    per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P_["per"].items()}
    sgt = {int(k): v[:2] for k, v in json.load(open(f"results/volume/reference/{CLIP}/ball_gt.json")).items() if v}
    b4 = [(m["frame"], m["x"], m["y"]) for m in json.load(open("results/ball/b4/key.json"))["moments"] if m["verdict"] == "ball"]
    wasb = load(sorted(glob.glob(f"results/volume/cache/{CLIP}/ball_cands_wasb_*_thr0.05.pkl"))[0])
    k34 = [(f, g[0], g[1]) for f, g in sgt.items()]
    grid = grid or list(itertools.product([1.5, 2.5, 3.5], [0.5, 0.75, 1.0], [1.0, 1.5]))
    out = {"grid": "conf_w x near_w x rf_weight(wy)", "finders": {}}
    for tag in ("old", "new"):
        rf = load(f"{RUN}/cands_{tag}_{CLIP}.pkl")
        row = {"alone_top_34": top_hits(rf, k34), "alone_top_b4": top_hits(rf, b4), "of": [len(k34), len(b4)], "picker": {}}
        for cw, nw, wy in grid:
            ball = BL.bridge(BL.pick_v2(BL.fuse_candidates(rf, wasb, wy=wy), H, L, W, per=per, fps=fps, conf_w=cw, near_w=nw, log=lambda *a: None), fps)
            row["picker"][f"{cw}/{nw}/{wy}"] = picker_hits(ball, sgt, b4)
        best = max(row["picker"].items(), key=lambda kv: (kv[1][0] + kv[1][1] / 10))
        row["default_2.5/0.75/1.5"] = row["picker"].get("2.5/0.75/1.5"); row["best"] = best
        out["finders"][tag] = row; print(tag, {k: v for k, v in row.items() if k != "picker"}, flush=True)
    return out
if __name__ == "__main__":
    o = main(); os.makedirs("results/ball/s3b", exist_ok=True); json.dump(o, open("results/ball/s3b/grade.json", "w"), indent=1)
