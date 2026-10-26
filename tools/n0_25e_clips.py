"""N0-25e (26 Oct, local, free): the app clips' sequence labels (stats.sequences[]: reached_final_third, end_reason) on the
exact inputs (results/volume/cache/<clip>/picker_inputs.pkl), steps like ipanema/run.py, old rule (reach at the last
frame, no stoppage end) vs new (P.SEQ_REACH 'any', P.SEQ_STOP_S 2). Writes results/qa/n0-25e/clips.json and checks that
the 17 picture-sheet moments (results/qa/n0-25e/moments.json, made 25 Oct) are exactly the sequences whose labels change.
    PYTHONPATH=. python tools/n0_25e_clips.py [--write-moments]"""
import sys, os, json, pickle, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, possession as P
from ipanema.run import periods_direction

CLIPS = {"SFKBP1109_s1200": "results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl",
         "p15u-vs-aik-2026-09-21-bd09_s2520": "results/volume/cache/p15u-vs-aik-2026-09-21-bd09_s2520/picker_inputs.pkl"}
OUT = "results/qa/n0-25e"; q = lambda *a: None


def clip_inputs(path, clip):
    I = pickle.load(open(path, "rb")); fps, H, L, W = I["fps"], I["H"], I["L"], I["W"]
    per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in I["per"].items()}
    ball = BL.bridge(BL.pick_v2(I["cands"], H, L, W, per=per, fps=fps, log=q), fps)
    frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0); state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q)
    ar, conf = P.direction(state, ballm, log=q); rd = periods_direction(".", clip)
    if rd is not None: ar = {"A": bool(rd), "B": not rd}
    elif min(conf.values()) < 0.15: ar = P.direction_from_keepers(per, L, log=q) or P.direction_fallback(per, L, log=q)
    cstate = P.spell_state(state, fps, P.SPELL_TAKE_S, P.SPELL_JOIN_S)
    return dict(fps=fps, L=L, W=W, ballm=ballm, bspeed=bspeed, state=state, cstate=cstate, dstate=dstate, pinfo=pinfo, ar=ar)


def both(D):
    a = (D["cstate"], D["ballm"], D["bspeed"], D["fps"], D["L"], D["ar"])
    old = P.sequences(*a, **D["pinfo"]["seq"], reach="end", stop_s=0)
    sp = P.stoppage_spans(D["dstate"], D["ballm"], D["fps"], D["L"], D["W"])
    new0 = P.sequences(*a, **D["pinfo"]["seq"], stops=sp, reach="any", stop_s=2.0, reach_min_s=0.0)   # the rule the 25 Oct sheets were cut for
    new = P.sequences(*a, **D["pinfo"]["seq"], stops=sp, reach="any", stop_s=2.0, reach_min_s=0.5)   # the default after grading
    return old, new0, new


def main():
    rep = {}; changed = []
    for clip, path in CLIPS.items():
        D = clip_inputs(path, clip); old, new0, new = both(D); assert len(old) == len(new) == len(new0)
        sp = P.stoppage_spans(D["dstate"], D["ballm"], D["fps"], D["L"], D["W"])
        rep[clip] = {"attack_right": D["ar"], "sequences": len(new), "stoppages": len(sp),
                     "reached_old": {t: sum(s["reached_final_third"] for s in old if s["team"] == t) for t in "AB"},
                     "reached_new_min0": {t: sum(s["reached_final_third"] for s in new0 if s["team"] == t) for t in "AB"},
                     "reached_new": {t: sum(s["reached_final_third"] for s in new if s["team"] == t) for t in "AB"},
                     "end_old": dict(collections.Counter(s["end_reason"] for s in old)), "end_new": dict(collections.Counter(s["end_reason"] for s in new))}
        rep[clip]["changed_min0.5"] = [n["t_start"] for o, n in zip(old, new) if (o["reached_final_third"], o["end_reason"]) != (n["reached_final_third"], n["end_reason"])]
        for o, n in zip(old, new0):
            ch = [c for c, k in (("reach", "reached_final_third"), ("end", "end_reason")) if o[k] != n[k]]
            if ch: changed.append((clip, n["team"], n["t_start"], n["t_end"], ch))
        print(clip, json.dumps(rep[clip]))
    M = json.load(open(f"{OUT}/moments.json"))
    sheet = sorted((m["clip"], m["team"], m["t_start"], m["t_end"], m["change"]) for m in M)
    rep["sheets_match_changes"] = sorted(changed) == sheet
    print("changed sequences", len(changed), "picture-sheet moments", len(sheet), "same:", rep["sheets_match_changes"])
    json.dump(rep, open(f"{OUT}/clips.json", "w"), indent=1)


if __name__ == "__main__":
    main()
