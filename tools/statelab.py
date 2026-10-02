"""S8 (2 Oct): one de-flickered possession state for passes / turnovers / pressures. Free, local, on the exact app inputs.
Keys: who-has-the-ball 99 moments (SFK-BP clip), 28 graded passes (SFK-BP clip), spells per minute (both clips)."""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, possession as P, analytics as AN
q = lambda *a: None
def load(clip):
    P_ = pickle.load(open(f"results/volume/cache/{clip}/picker_inputs.pkl", "rb")); fps = P_["fps"]; H = P_["H"]; L, W = P_["L"], P_["W"]
    per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P_["per"].items()}
    ball = BL.bridge(BL.pick_v2(P_["cands"], H, L, W, per=per, fps=fps, log=q), fps); frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0)
    state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q); ar, _ = P.direction(state, ballm, log=q)
    return dict(per=per, ball=ball, ballm=ballm, frames_=frames_, state=state, bspeed=bspeed, dstate=dstate, pinfo=pinfo, ar=ar, fps=fps, L=L, W=W, H=H)
def grade_who(state, key):
    ok = tot = 0
    for a in key:
        want = {"dark": 0, "white": 1, "light": 1, "loose": 2}.get(a["truth"])
        if want is None or a["frame"] >= len(state): continue
        tot += 1; ok += int(state[a["frame"]] == want)
    return ok, tot
def flips(state):
    t = [int(s) for s in state if int(s) < 2]; return sum(1 for a, b in zip(t, t[1:]) if a != b)
def clean_state(state, fps, take_s, join_s, fill_loose=False):
    """per-frame state with the E5 take-over rule: a team keeps possession until the other has held the ball take_s longer;
    rejected takeovers read as the owner; loose frames inside a spell read as the owner only if fill_loose"""
    out = np.array(state, int).copy(); raw = np.array(state, int)
    for sq in P.sequence_spans(state, fps, take_s, join_s):
        a, b = sq["start"], sq["end"] + 1; t = 0 if sq["team"] == "A" else 1
        m = (raw[a:b] < 2) if not fill_loose else (raw[a:b] != 3)
        out[a:b][m] = t
    return out
def grade_passes(ctx, state_used, filt=True):
    c = ctx; st = c["state"] if state_used is None else state_used; tvs = P.turnovers(c["per"], c["frames_"], st, c["ballm"], c["fps"], c["ar"], 2.0, 5.0, min_before_s=c["pinfo"]["turnover_s"], min_after_s=c["pinfo"]["turnover_s"])
    ln = AN.lanes(c["per"], c["frames_"], c["ar"], 1.5, 30.0); ps, _ = AN.passes(c["per"], c["frames_"], tvs, ln, c["ar"], c["fps"], state=(state_used if filt else None))
    exp = json.load(open("results/volume/runs/matches/SFKBP1109_s1200/stats.json"))["passes"]; A = json.load(open("results/review/passcheck_answers.json"))
    fake = set(A["fake"]); unsure = set(A["unsure"]); graded = [i for i in range(A["graded_first_n"]) if i not in unsure]
    keep = [any(abs(p["t"] - exp[i]["t"]) < 0.3 for p in ps) for i in graded]
    rk = sum(k for i, k in zip(graded, keep) if i not in fake); fk = sum(k for i, k in zip(graded, keep) if i in fake)
    return len(ps), rk, len(graded) - len(fake), fk, len(fake), len(tvs)
if __name__ == "__main__":  # noqa
    key = json.load(open("results/review/who_answers.json"))["moments"]
    S = load("SFKBP1109_s1200"); A_ = load("p15u-vs-aik-2026-09-21-bd09_s2520"); mins = len(S["per"]) / S["fps"] / 60
    print("raw state: who", grade_who(S["state"], key), "| team flips/min SFK %.0f AIK %.0f" % (flips(S["state"]) / mins, flips(A_["state"]) / mins))
    print("passes with raw state (no state filter): n, real kept, fake kept, turnovers =", grade_passes(S, None, filt=False))
    print("passes with raw state as filter:", grade_passes(S, S["state"]))
    for take, join, fill in ((0.5, 3.0, False), (0.5, 3.0, True), (1.0, 3.0, False), (1.0, 3.0, True), (1.5, 3.0, False), (2.0, 3.0, False), (1.0, 5.0, False)):
        cs = clean_state(S["state"], S["fps"], take, join, fill); ca = clean_state(A_["state"], A_["fps"], take, join, fill)
        print(f"take {take} join {join} fill_loose {fill}: who {grade_who(cs, key)} | flips/min SFK {flips(cs) / mins:.0f} AIK {flips(ca) / mins:.0f} | passes {grade_passes(S, cs)}")
    print("--- which graded passes survive, take 1.5 / join 3 (R=real F=fake, + kept, - lost); state at pass time, passer team")
    cs = clean_state(S["state"], S["fps"], 1.5, 3.0); c = S
    tvs = P.turnovers(c["per"], c["frames_"], cs, c["ballm"], c["fps"], c["ar"], 2.0, 5.0, min_before_s=c["pinfo"]["turnover_s"], min_after_s=c["pinfo"]["turnover_s"])
    ln = AN.lanes(c["per"], c["frames_"], c["ar"], 1.5, 30.0); ps, _ = AN.passes(c["per"], c["frames_"], tvs, ln, c["ar"], c["fps"], state=cs)
    exp = json.load(open("results/volume/runs/matches/SFKBP1109_s1200/stats.json"))["passes"]; A = json.load(open("results/review/passcheck_answers.json"))
    for i in range(30):
        if i in A["unsure"]: continue
        k = int(round(exp[i]["t"] * c["fps"])); win = [int(cs[j]) for j in range(max(0, k - 15), min(len(cs), k + 16))]
        kept = any(abs(p["t"] - exp[i]["t"]) < 0.3 for p in ps)
        print(i, "F" if i in A["fake"] else "R", "+" if kept else "-", "team", exp[i]["team"], "state -0.5..+0.5 s:", "".join("AB.x"[s] for s in win))
    print("--- key mapping check: dark->A (as graded) vs dark->B")
    def grade_sw(state, key):
        ok = tot = 0
        for a in key:
            want = {"dark": 1, "white": 0, "light": 0, "loose": 2}.get(a["truth"])
            if want is None or a["frame"] >= len(state): continue
            tot += 1; ok += int(state[a["frame"]] == want)
        return ok, tot
    for name, st in (("raw", S["state"]), ("clean 1.5/3", cs)):
        lo = sum(1 for a in key if a["truth"] == "loose" and st[a["frame"]] == 2)
        print(name, "dark=A:", grade_who(st, key), "dark=B:", grade_sw(st, key), "| loose right:", lo, "/ 41")
    # which kit is A on these inputs? mean pixel brightness is not in the inputs; use the export's kit files instead
    import os; print("kit files:", [f for f in os.listdir("results/volume/runs/matches/SFKBP1109_s1200") if f.startswith("kit")])

def carriers_px(per, ball, H, state, carrier_m=2.5, near_r=5.0):
    """S8: the carrier is the nearest player (in the picture, side-to-side scale at the ball, like possession_simple) of
    the team the state gives; no carrier when the state is loose/dead. pressure_m / near_opps in pitch metres as before."""
    from ipanema.calibration import to_m
    n = len(per); frames_ = []; ballm = {}
    for k in range(n):
        rec = {"frame": k, "carrier": None, "team": None, "pressure_m": None, "near_opps": None, "pos": None}
        if k in ball and H[k] is not None:
            b = ball[k]; bm = to_m(H[k], [b])[0]; ballm[k] = bm; st = int(state[k]); tm = "AB"[st] if st < 2 else None
            rows = [r for r in per[k] if r[3] is not None and r[1] == tm] if tm else []
            if rows:
                try:
                    inv = np.linalg.inv(H[k]); p = [inv @ np.array([b[0] + dx, b[1], 1.0]) for dx in (0.0, 1.0)]; s_h = float(np.linalg.norm(p[1][:2] / p[1][2] - p[0][:2] / p[0][2]))
                except Exception: s_h = None
                if s_h:
                    d = [np.hypot(r[3][0] - b[0], r[3][1] - b[1]) * s_h for r in rows]; ci = int(np.argmin(d))
                    if d[ci] <= carrier_m:
                        tid, pos = rows[ci][0], rows[ci][2]; opps = [r[2] for r in per[k] if r[1] != tm]; od = [np.linalg.norm(o - pos) for o in opps]
                        rec.update({"carrier": tid, "team": tm, "pos": pos.tolist(), "pressure_m": round(float(min(od)), 2) if od else None, "near_opps": int(sum(x <= near_r for x in od))})
        frames_.append(rec)
    return frames_, ballm
def grade_passes2(c, state_used, frames_, filt=False):
    tvs = P.turnovers(c["per"], frames_, state_used, c["ballm"], c["fps"], c["ar"], 2.0, 5.0, min_before_s=c["pinfo"]["turnover_s"], min_after_s=c["pinfo"]["turnover_s"])
    ln = AN.lanes(c["per"], frames_, c["ar"], 1.5, 30.0); ps, _ = AN.passes(c["per"], frames_, tvs, ln, c["ar"], c["fps"], state=(state_used if filt else None))
    exp = json.load(open("results/volume/runs/matches/SFKBP1109_s1200/stats.json"))["passes"]; A = json.load(open("results/review/passcheck_answers.json"))
    fake = set(A["fake"]); unsure = set(A["unsure"]); graded = [i for i in range(A["graded_first_n"]) if i not in unsure]
    keep = [any(abs(p["t"] - exp[i]["t"]) < 0.3 for p in ps) for i in graded]
    rk = sum(k for i, k in zip(graded, keep) if i not in fake); fk = sum(k for i, k in zip(graded, keep) if i in fake)
    return dict(passes=len(ps), real_kept=f"{rk}/{len(graded) - len(fake)}", fake_kept=f"{fk}/{len(fake)}", turnovers=len(tvs))
if __name__ == "__main__":  # noqa
    print("--- carriers from the state (pixel rule), passes without the state filter")
    for take in (0.0, 1.0, 1.5):
        for cm in (2.5, 1.5):
            cs = S["state"] if take == 0 else clean_state(S["state"], S["fps"], take, 3.0)
            fr, _ = carriers_px(S["per"], S["ball"], S["H"], cs, carrier_m=cm)
            print(f"take {take} carrier_m {cm}:", grade_passes2(S, cs, fr))
