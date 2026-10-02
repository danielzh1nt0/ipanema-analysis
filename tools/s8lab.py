"""S8 (2 Oct): one de-flickered possession state for the counted stats. The raw state (possession_simple) flips 35-41x per
minute; sequences already ignore short takeovers (E5 spans). Here: build a 'spell state' from the E5 spans (team inside a
span, loose/dead outside) and feed it to turnovers, passes and pressures, on the exact app inputs of both clips. The raw
state stays for the ball/possession display (82/99 moments). Measures: spells/min, passes (and the 28 graded SFK-BP ones:
real kept / fake kept), turnovers, pressures per 90, sequences. Free, local.
    PYTHONPATH=. python tools/s8lab.py"""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, possession as P, analytics as AN
q = lambda *a: None

def spell_state(state, fps, take_s, join_s):
    """team per frame from the E5 spans (0 A, 1 B), else the raw loose/dead value"""
    out = np.where(np.asarray(state) < 2, 2, np.asarray(state)).copy()
    for sq in P.sequence_spans(state, fps, take_s, join_s): out[sq["start"]:sq["end"] + 1] = 0 if sq["team"] == "A" else 1
    return out

def load(clip):
    P_ = pickle.load(open(f"results/volume/cache/{clip}/picker_inputs.pkl", "rb")); fps = P_["fps"]; H = P_["H"]; L, W = P_["L"], P_["W"]
    per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P_["per"].items()}
    ball = BL.bridge(BL.pick_v2(P_["cands"], H, L, W, per=per, fps=fps, log=q), fps); frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0)
    state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q); ar, _ = P.direction(state, ballm, log=q)
    return dict(per=per, frames_=frames_, ballm=ballm, state=state, bspeed=bspeed, pinfo=pinfo, ar=ar, fps=fps, L=L, W=W)

def run(c, st, pass_state=False):
    per, frames_, ballm, fps, L, W, ar = c["per"], c["frames_"], c["ballm"], c["fps"], c["L"], c["W"], c["ar"]
    tvs = P.turnovers(per, frames_, st, ballm, fps, ar, 2.0, 5.0, min_before_s=c["pinfo"]["turnover_s"], min_after_s=c["pinfo"]["turnover_s"])
    ln = AN.lanes(per, frames_, ar, 1.5, 30.0)
    ps, tracks = AN.passes(per, frames_, tvs, ln, ar, fps, state=(st if pass_state else None))
    seqs = P.sequences(st, ballm, c["bspeed"], fps, L, ar, **c["pinfo"]["seq"])
    stt = AN.stats(per, frames_, tvs, ps, tracks, st, fps, L, W, ar, 2.0, 5.0, sequences_=seqs)
    flips = int(sum(1 for i in range(1, len(st)) if st[i] < 2 and st[i - 1] < 2 and st[i] != st[i - 1]))
    return dict(passes=ps, turnovers=len(tvs), seqs=len(seqs), flips=flips, press={t["team"]: t["pressures_applied"] for t in stt["teams"]}, ttp={t["team"]: t["time_to_press_median_s"] for t in stt["teams"]})

if __name__ == "__main__":
    A = json.load(open("results/review/passcheck_answers.json")); fake = set(A["fake"]); unsure = set(A["unsure"]); graded = [i for i in range(A["graded_first_n"]) if i not in unsure]
    exp = json.load(open("results/volume/runs/matches/SFKBP1109_s1200/stats.json"))["passes"]
    def grade(ps):
        keep = [any(abs(p["t"] - exp[i]["t"]) < 0.3 for p in ps) for i in graded]
        return sum(k for i, k in zip(graded, keep) if i not in fake), sum(k for i, k in zip(graded, keep) if i in fake)
    out = {}
    for clip in ("SFKBP1109_s1200", "p15u-vs-aik-2026-09-21-bd09_s2520"):
        c = load(clip); mins = len(c["per"]) / c["fps"] / 60; print(clip, f"{mins:.1f} min")
        trials = [("raw state (today)", c["state"], False)]
        for take, join in ((0.5, 3.0), (1.0, 3.0), (1.5, 3.0), (1.0, 5.0)):
            trials.append((f"spell state take {take} join {join}", spell_state(c["state"], c["fps"], take, join), False))
            trials.append((f"  + passes filtered by it", spell_state(c["state"], c["fps"], take, join), True))
        for name, st, pst in trials:
            r = run(c, st, pst); g = grade(r["passes"]) if clip.startswith("SFK") else None
            print(f"  {name:36s} flips/min {r['flips'] / mins:5.1f} | passes {len(r['passes']):3d}" + (f" (real kept {g[0]}/20, fake kept {g[1]}/8)" if g else "") + f" | turnovers {r['turnovers']:3d} | seqs {r['seqs']:3d} | pressures/90 A {r['press']['A'] * 90 / mins:4.0f} B {r['press']['B'] * 90 / mins:4.0f} | time to press {r['ttp']}")
            out[f"{clip}|{name.strip()}"] = {"flips_per_min": r["flips"] / mins, "passes": len(r["passes"]), "graded": g, "turnovers": r["turnovers"], "sequences": r["seqs"], "pressures_per_90": {k: v * 90 / mins for k, v in r["press"].items()}, "ttp": r["ttp"]}
    json.dump(out, open("results/possession/s8lab_2026-10-02.json", "w"), indent=1)
