"""N0-20b (20 Oct): final-third entries on the SFK-BP test clip, replayed from the app's own picker inputs.
old = today's rule (raw possession state, any crossing of the 2/3 line); new = options under test.
    PYTHONPATH=. python tools/entrylab.py  -> results/qa/n0-20b/entrylab.json"""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, possession as P, metrics as M
from ipanema.run import periods_direction
q = lambda *a: None; CLIP = "SFKBP1109_s1200"; OUT = "results/qa/n0-20b"

def replay():
    P_ = pickle.load(open(f"results/volume/cache/{CLIP}/picker_inputs.pkl", "rb")); fps, H, L, W = P_["fps"], P_["H"], P_["L"], P_["W"]
    per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P_["per"].items()}
    ball = BL.bridge(BL.pick_v2(P_["cands"], H, L, W, per=per, fps=fps, log=q), fps)
    frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0); state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q)
    ad = periods_direction(".", CLIP); ar = {"A": bool(ad), "B": not ad} if ad is not None else P.direction(state, ballm, log=q)[0]
    return dict(fps=fps, L=L, state=state, cstate=P.spell_state(state, fps), ballm=ballm, bspeed=bspeed, ar=ar)

def moments(R=None, gap_s=2.0):
    """clusters of candidate entries (union of all rules, entries < gap_s apart merged) with what the free-runner sheet
    draws: the ball (app picker, px) every 0.5 s and the attacking team's 2/3 line in pixels. Which rule counted what is
    written to a separate key, opened only after grading."""
    P_ = pickle.load(open(f"results/volume/cache/{CLIP}/picker_inputs.pkl", "rb")); fps, H, L, W = P_["fps"], P_["H"], P_["L"], P_["W"]
    per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P_["per"].items()}
    ball = BL.bridge(BL.pick_v2(P_["cands"], H, L, W, per=per, fps=fps, log=q), fps)
    lab = json.load(open(f"{OUT}/entrylab.json")); ar = lab["attack_right"]
    rules = [k for k, v in lab.items() if isinstance(v, dict) and isinstance(v.get("A"), dict)]
    cands = sorted({(tm, t) for r in rules for tm in "AB" for t in lab[r][tm]["entries"]})
    cl = []
    for tm, t in cands:
        if cl and cl[-1]["team"] == tm and t - cl[-1]["t1"] < gap_s: cl[-1]["t1"] = t; cl[-1]["ts"].append(t)
        else: cl.append({"team": tm, "t0": t, "t1": t, "ts": [t]})
    def line_px(k, tm):
        """the 2/3 line of the team's attack, metres (0..L, 0..W) -> pixels through the frame's camera (H maps metres -> px)"""
        Hk = H.get(k)
        if Hk is None or hasattr(Hk, "to_m"): return None
        x = 2 * L / 3 if ar[tm] else L / 3; out = []
        for y in (0.0, W):
            p = np.asarray(Hk, float) @ np.array([x, y, 1.0]); out.append([round(float(p[0] / p[2]), 1), round(float(p[1] / p[2]), 1)])
        return out
    key = {}; out = []
    for i, c in enumerate(cl):
        mid = f"m{i + 1:02d}"; ts = [round(c["t0"] - 2 + 0.5 * j, 2) for j in range(int((c["t1"] - c["t0"] + 3.5) / 0.5) + 1)][:20]
        tiles = []
        for t in ts:
            k = int(round(t * fps)); b = ball.get(k)
            tiles.append({"t": t, "frame": k, "ball_px": [round(float(b[0]), 1), round(float(b[1]), 1)] if b is not None else None, "line_px": line_px(k, c["team"])})
        out.append({"id": mid, "team": "dark" if c["team"] == "A" else "light", "attacks": "right" if ar[c["team"]] else "left", "tiles": tiles})
        key[mid] = {"team": c["team"], "candidate_times": c["ts"], "counted_by": {r: [t for t in c["ts"] if t in lab[r][c["team"]]["entries"]] for r in rules}}
    json.dump({"clip": CLIP, "fps": fps, "moments": out}, open(f"{OUT}/moments.json", "w"), indent=1)
    json.dump(key, open(f"{OUT}/moments_key.json", "w"), indent=1)
    print(len(out), "moments ->", f"{OUT}/moments.json")

if __name__ == "__main__" and sys.argv[1:2] == ["moments"]: moments()
elif __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); R = replay(); fps, L = R["fps"], R["L"]
    out = {"attack_right": R["ar"], "frames": len(R["state"])}
    for name, kw in {"old (app today)": dict(state=R["state"]), "spell state only": dict(state=R["cstate"]),
                     "raw state + re-arm 3 m / 1 s": dict(state=R["state"], rearm_m=3.0, rearm_s=1.0),
                     "spell state + re-arm 3 m / 1 s": dict(state=R["cstate"], rearm_m=3.0, rearm_s=1.0)}.items():
        st = kw.pop("state"); r = M.field_tilt_and_entries(st, R["ballm"], R["bspeed"], fps, L, R["ar"], **kw)
        out[name] = {tm: {"tilt": r[tm]["field_tilt_pct"], "entries": [e["t"] for e in r[tm]["final_third_entries"]]} for tm in "AB"}
        print(f"{name:34s}", {tm: (r[tm]["field_tilt_pct"], r[tm]["entries_count"]) for tm in "AB"})
    json.dump(out, open(f"{OUT}/entrylab.json", "w"), indent=1)
