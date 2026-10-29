"""N0-29d: the runs' 'with the ball' label on the two app clips (exact picker inputs, replayed like tools/entrylab.py):
how many runs get a label (not None) and how many labels change, app (raw state) vs spell state at the start vs
spell + 3-s look-back. -> results/qa/n0-29d/clips.json
    PYTHONPATH=. python tools/n0_29d_clips.py [--out results/qa/n0-29d]"""
import sys, os, json, pickle, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import ball as BL, possession as P, metrics as M
from ipanema.run import periods_direction
import n0_29d_runball as R

q = lambda *a: None
CLIPS = ("SFKBP1109_s1200", "p15u-vs-aik-2026-09-21-bd09_s2520")


def replay(clip):
    d = pickle.load(open(f"results/volume/cache/{clip}/picker_inputs.pkl", "rb")); fps, H, L, W = d["fps"], d["H"], d["L"], d["W"]
    per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] + list(r[6:]) for r in v] for k, v in d["per"].items()}
    ball = BL.bridge(BL.pick_v2(d["cands"], H, L, W, per=per, fps=fps, log=q), fps)
    frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0); state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q)
    ad = periods_direction(".", clip); ar = {"A": bool(ad), "B": not ad} if ad is not None else P.direction(state, ballm, log=q)[0]
    return per, fps, np.asarray(state), np.asarray(P.spell_state(state, fps)), ar


def main(out):
    res = {}
    for c in CLIPS:
        per, fps, st, cst, ar = replay(c); rs = M.runs(per, fps, st, ar); rows = []
        for r in rs:
            a, b = int(round(r["t"] * fps)), int(round(r["t_end"] * fps))
            rows.append({"player": r["player"], "team": r["team"], "t": r["t"], "app": r["with_ball"], **{k: R.label(k, st, cst, a, b, r["team"], fps) for k in R.LABELS}})
        assert all(x["app"] == x["raw_start"] for x in rows)
        res[c] = {"runs": len(rows), **{k: dict(collections.Counter(str(x[k]) for x in rows)) for k in R.LABELS},
                  "changed_vs_app": {k: sum(x["app"] is not None and x[k] is not None and x[k] != x["app"] for x in rows) for k in R.LABELS},
                  "rows": rows}
        print(c, json.dumps({k: v for k, v in res[c].items() if k != "rows"}))
    os.makedirs(out, exist_ok=True); json.dump(res, open(f"{out}/clips.json", "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]; main(a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-29d")
