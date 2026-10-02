"""S4 (2 Oct): reproduce the SFK-BP clip's exported passes from the exact app inputs, then test pass-rule changes against
the by-eye verdicts on the first 30 (results/review/passcheck_answers.json). Free, local."""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, possession as P, analytics as AN, tracking as TR
P_ = pickle.load(open("results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl", "rb")); fps = P_["fps"]; H = P_["H"]; L, W = P_["L"], P_["W"]
per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P_["per"].items()}
q = lambda *a: None
ball = BL.bridge(BL.pick_v2(P_["cands"], H, L, W, per=per, fps=fps, log=q), fps)
frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0)
state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q)
attack_right, conf = P.direction(state, ballm, log=q)
tvs = P.turnovers(per, frames_, state, ballm, fps, attack_right, 2.0, 5.0, min_before_s=pinfo["turnover_s"], min_after_s=pinfo["turnover_s"])
ln = AN.lanes(per, frames_, attack_right, 1.5, 30.0)
def run(**kw):
    ps, _ = AN.passes(per, frames_, tvs, ln, attack_right, fps, **kw); return ps
base = run(); print("passes reproduced:", len(base), "(app exported 87)")
exp = json.load(open("results/volume/runs/matches/SFKBP1109_s1200/stats.json"))["passes"]
print("same times as the export:", sum(1 for a, b in zip(base, exp) if abs(a["t"] - b["t"]) < 0.05), "/", min(len(base), len(exp)))
A = json.load(open("results/review/passcheck_answers.json"))
fake = set(A["fake"]); unsure = set(A["unsure"]); graded = [i for i in range(A["graded_first_n"]) if i not in unsure]
def grade(ps, name):
    # map each exported pass index to a time; a rule 'keeps' the pass if a pass within 0.3 s remains
    keep = [any(abs(p["t"] - exp[i]["t"]) < 0.3 for p in ps) for i in graded]
    real_kept = sum(k for i, k in zip(graded, keep) if i not in fake); fake_kept = sum(k for i, k in zip(graded, keep) if i in fake)
    print(f"{name:40s} passes {len(ps):3d} | of graded: real kept {real_kept}/{len(graded) - len(fake)}, fake kept {fake_kept}/{len(fake)}")
grade(base, "current rule")
for kw in ({"ball_min_m": 5.0}, {"ball_min_m": 8.0}, {"ball_min_m": 5.0, "dedupe_s": 1.0}, {"ball_min_m": 8.0, "dedupe_s": 1.0}, {"ball_min_m": 5.0, "dedupe_s": 1.0, "state": state}, {"ball_min_m": 8.0, "dedupe_s": 1.5, "state": state}):
    try: grade(run(ballm=ballm, **kw), ", ".join(f"{k}={'yes' if k == 'state' else v}" for k, v in kw.items()))
    except TypeError as e: print("not implemented:", e); break
# why do the fakes survive? per graded pass: ball displacement between contacts, peak ball speed in the window, state, gap
ps, _ = AN.passes(per, frames_, tvs, ln, attack_right, fps, debug=True)
def bstat(e):
    a0, a1, b0, b1 = e; ks = [j for j in range(a1, b0 + 1) if j in ballm]
    if len(ks) < 2: return None
    pts = np.array([ballm[j] for j in ks]); d = np.linalg.norm(pts[-1] - pts[0]); sp = np.linalg.norm(np.diff(pts, axis=0), axis=1) * fps / np.maximum(1, np.diff(ks))
    return round(float(d), 1), round(float(np.percentile(sp, 90)), 1), int(state[a1]), round((b0 - a1) / fps, 2), round((a1 - a0) / fps, 2), round((b1 - b0) / fps, 2)
print("idx fake? | ball moved m, ball speed p90 m/s, state, gap s, touch A s, touch B s")
for i in graded: print(i, "FAKE" if i in fake else "real", bstat(ps[i]["_eps"]))
print("--- pixel features: share of window frames where the ball px stays within 40 px of the passer; ball off pitch at either contact")
pxpos = {k: {r[0]: r[3] for r in v if r[3] is not None} for k, v in per.items()}
def pxstat(p):
    a0, a1, b0, b1 = p["_eps"]; ks = [j for j in range(a1, b0 + 1) if j in ball and p["from"] in pxpos.get(j, {})]
    if not ks: return None
    near = np.mean([np.hypot(ball[j][0] - pxpos[j][p["from"]][0], ball[j][1] - pxpos[j][p["from"]][1]) < 40 for j in ks])
    offp = any(not (-2 <= ballm[j][0] <= L + 2 and -2 <= ballm[j][1] <= W + 2) for j in (a1, b0) if j in ballm)
    return round(float(near), 2), offp, len(ks)
for i in graded: print(i, "FAKE" if i in fake else "real", pxstat(ps[i]))
print("--- camera-corrected ball travel in px over the window (ball px minus median shift of all players), and dead/stoppage")
def travel(p):
    a0, a1, b0, b1 = p["_eps"]; ks = [j for j in range(max(0, a1 - 3), min(len(per), b0 + 4)) if j in ball]
    if len(ks) < 2: return None
    cum = np.zeros(2); rel = [np.array(ball[ks[0]], float)]
    for j0, j1 in zip(ks[:-1], ks[1:]):
        d = [np.asarray(pxpos[j1][t]) - np.asarray(pxpos[j0][t]) for t in pxpos.get(j1, {}) if t in pxpos.get(j0, {})]
        cam = np.median(d, 0) if len(d) >= 3 else np.zeros(2); cum += cam
        rel.append(np.array(ball[j1], float) - cum)
    rel = np.array(rel); return round(float(np.linalg.norm(rel[-1] - rel[0])), 0), round(float(np.abs(np.diff(rel, axis=0)).sum()), 0), int(dstate[a1]) if a1 < len(dstate) else None
for i in graded: print(i, "FAKE" if i in fake else "real", travel(ps[i]))
