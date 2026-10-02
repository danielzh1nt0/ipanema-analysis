"""F1 (2 Oct): why the full SFK-BP match scores the ball 25/34 when the 5-min clip scores 29/34. Free, offline, on the
EXACT app inputs of the clip (results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl). The clip starts at 1200 s = the
first frame of full-match piece 4, so the full run sees the same frames, the same line calibration and the same finders.
The one rule the full run adds: frames the line calibration is unsure about are blanked (no players, no ball guesses;
fullmatch.apply_unknown). This replays that rule on the clip inputs and scores the 34 checked moments + the B4 key.
    python tools/f1_sim.py   -> results/ball/f1/sim.json"""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, linecal as LC, fullmatch as FM

q = lambda *a: None
on = lambda b, f, x, y: f in b and np.hypot(b[f][0] - x, b[f][1] - y) <= 30

def load(m="SFKBP1109_s1200"):
    P = pickle.load(open(f"results/volume/cache/{m}/picker_inputs.pkl", "rb"))
    P["per"] = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P["per"].items()}
    return P

def variant(P, unsure, blank_players, blank_cands):
    ok = np.ones(len(P["cands"]), bool); ok[[k for k in unsure if k < len(ok)]] = False
    per = {k: ([] if (blank_players and not ok[k]) else v) for k, v in P["per"].items()}
    cands = {k: ([] if (blank_cands and not ok[k]) else v) for k, v in P["cands"].items()}
    return BL.bridge(BL.pick_v2(cands, P["H"], P["L"], P["W"], per=per, fps=P["fps"], log=q), P["fps"]), cands, per

def stats(P, per, ball):
    """possession etc. as run.analyse computes them after the pick (tools/e4lab.py), graded on the 99 who-has-the-ball moments"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("e4lab", os.path.join(os.path.dirname(os.path.abspath(__file__)), "e4lab.py")); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
    n = len(per); c = dict(per={k: per[k] for k in range(n)}, ball=ball, H=[P["H"][k] for k in range(n)], L=P["L"], W=P["W"], fps=P["fps"], boxh=None,
                           key=json.load(open("results/review/who_answers.json"))["moments"])
    r = E.stats(c, "simple")
    return {k: r[k] for k in ("who_has_ball_right", "possession_pct_dark_A", "loose_pct", "dead_pct", "sequences", "turnovers", "passes", "restarts")}

def main(out="results/ball/f1/sim.json", with_stats=True):
    P = load()
    gt = {int(k): v for k, v in json.load(open("results/volume/reference/SFKBP1109_s1200/ball_gt.json")).items() if v}
    b4 = [m for m in json.load(open("results/ball/b4/key.json"))["moments"] if m["verdict"] == "ball"]
    cal = LC.calibration_for_clip("calibration/SFKBP1109_lines_match.json", len(P["cands"]), P["fps"], 1920, 1080, offset_s=1200.0, log=q)
    unsure = cal["unsure"]
    rows = {}
    for name, bp, bc in (("clip (nothing blanked)", False, False), ("full match now (players + ball guesses blanked)", True, True),
                         ("ball guesses kept, players blanked", True, False), ("players kept, ball guesses blanked", False, True)):
        ball, cands, per = variant(P, unsure, bp, bc)
        hit = {f: bool(on(ball, f, *g[:2])) for f, g in gt.items()}
        ceil = sum(any(np.hypot(x - g[0], y - g[1]) <= 30 for x, y, _ in cands.get(f, [])) for f, g in gt.items())
        rows[name] = {"sfk34": int(sum(hit.values())), "ceiling": int(ceil), "b4": int(sum(on(ball, m["frame"], m["x"], m["y"]) for m in b4)),
                      "missed": sorted(f for f, h in hit.items() if not h), "full_frame_of_missed": [f + int(round(1200 * P["fps"])) for f in sorted(f for f, h in hit.items() if not h)],
                      **({"stats": stats(P, per, ball)} if with_stats else {})}
        print(name, {k: v for k, v in rows[name].items() if k != "full_frame_of_missed"}, flush=True)
    res = {"unsure_frames": len(unsure), "of": len(P["cands"]), "key_moments_on_unsure_frames": sorted(f for f in gt if f in unsure),
           "b4_of": len(b4), "variants": rows}
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, "w"), indent=1)
    return res

if __name__ == "__main__": main()
