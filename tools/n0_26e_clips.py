"""N0-26e (26 Oct, local, free): our shot detector on the app clips' exact inputs (as n0_25e_clips), old (play must turn
'dead' in the possession state, which possession_simple never does -> only goal-mouth crossings) vs new default (goal =
next restart from the centre spot) vs option IPANEMA_SHOT_END=stops (the restart step's stoppages end a shot too), next to Veo's own shot list for the clip
(ipanema/run.veo_source). -> results/qa/n0-26e/clips.json
    PYTHONPATH=. python tools/n0_26e_clips.py"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P, metrics as M, veo as VEO, run as R
import n0_25e_clips as C

OUT = "results/qa/n0-26e"


def main():
    res = {}
    for clip, path in C.CLIPS.items():
        D = C.clip_inputs(path, clip); fps, L, W = D["fps"], D["L"], D["W"]
        rst = P.restarts(D["dstate"], D["ballm"], fps, L, W)
        sp = P.stoppage_spans(D["dstate"], D["ballm"], fps, L, W)
        a = (D["state"], D["ballm"], D["bspeed"], fps, L, W, D["ar"], rst)
        old = M.shots(*a, stops=sp, end="state", goal_rule="unknown"); new = M.shots(*a, stops=sp, end="state", goal_rule="centre"); opt = M.shots(*a, stops=sp, end="stops", goal_rule="centre")
        dur = len(D["state"]) / fps; src = R.veo_source(".", clip, dur, L, W, log=lambda *x: None)
        veo = []
        if src:
            veo = [{"t": float(t), "kind": k} for t, k in src[0]]
        sc = {n: VEO.score_detector(s, [v for v in veo]) if veo else None for n, s in (("old", old), ("new", new), ("stops", opt))}
        res[clip] = {"minutes": round(dur / 60, 1), "stoppages": len(sp), "veo": veo, "old": old, "new": new, "stops": opt, "score_vs_veo": sc}
        print(clip, json.dumps({k: v for k, v in res[clip].items() if k not in ("old", "new", "stops")}), "old", [(s["t"], s["team"], s["outcome"]) for s in old], "new", [(s["t"], s["team"], s["outcome"]) for s in new], "stops", [(s["t"], s["team"], s["outcome"]) for s in opt], flush=True)
    os.makedirs(OUT, exist_ok=True); json.dump(res, open(f"{OUT}/clips.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    main()
