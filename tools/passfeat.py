import sys, os, json, pickle, numpy as np
sys.path.insert(0, "/home/claude/ipanema-analysis")
from ipanema import ball as BL, possession as P, analytics as AN
P_ = pickle.load(open("results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl", "rb")); fps = P_["fps"]; H = P_["H"]; L, W = P_["L"], P_["W"]
per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P_["per"].items()}
q = lambda *a: None
raw = BL.pick_v2(P_["cands"], H, L, W, per=per, fps=fps, log=q); ball = BL.bridge(raw, fps)
frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0)
state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q)
attack_right, conf = P.direction(state, ballm, log=q)
tvs = P.turnovers(per, frames_, state, ballm, fps, attack_right, 2.0, 5.0, min_before_s=pinfo["turnover_s"], min_after_s=pinfo["turnover_s"])
ln = AN.lanes(per, frames_, attack_right, 1.5, 30.0)
ps, _ = AN.passes(per, frames_, tvs, ln, attack_right, fps, debug=True)
A = json.load(open("results/review/passcheck_answers.json")); fake = set(A["fake"]); unsure = set(A["unsure"])
cands = P_["cands"]
def confof(k):
    if k not in raw: return None
    x, y = raw[k][0], raw[k][1]
    best = [c for c in cands.get(k, []) if abs(c[0]-x) < 2 and abs(c[1]-y) < 2]
    return best[0][2] if best else None
print("idx verdict | window s | seen share | bridged share | mean conf | max 1-frame jump m | n jumps>3m | path/straight")
for i in range(30):
    if i in unsure: continue
    a0, a1, b0, b1 = ps[i]["_eps"]; ks = list(range(a0, b1 + 1))
    seen = [k for k in ks if k in raw]; br = [k for k in ks if k in ball and k not in raw]
    cf = [confof(k) for k in seen]; cf = [c for c in cf if c is not None]
    mk = [k for k in ks if k in ballm]; pts = np.array([ballm[k] for k in mk]) if len(mk) > 1 else None
    if pts is not None:
        st = np.linalg.norm(np.diff(pts, axis=0), axis=1); jm = st.max(); nj = int((st > 3).sum()); ratio = st.sum() / max(0.1, np.linalg.norm(pts[-1] - pts[0]))
    else: jm = nj = ratio = None
    print(i, "FAKE" if i in fake else "real", "|", round(len(ks)/fps, 2), "|", round(len(seen)/len(ks), 2), "|", round(len(br)/len(ks), 2), "|", round(float(np.mean(cf)), 2) if cf else None, "|", None if jm is None else round(float(jm), 1), "|", nj, "|", None if ratio is None else round(float(ratio), 1))

print("--- stationary candidate: share of window frames with a candidate within R of one camera-corrected point (best point), conf of that cluster")
def shift(k0, k):
    # camera shift between frames: median player pixel shift for players seen in both
    a = {r[0]: r[3] for r in per.get(k0, []) if r[3] is not None}; b = {r[0]: r[3] for r in per.get(k, []) if r[3] is not None}
    d = [b[i] - a[i] for i in a if i in b]
    return np.median(d, axis=0) if len(d) >= 3 else None
def stationary(a0, b1, R=30, min_conf=0.15):
    ks = [k for k in range(a0, b1 + 1) if k in cands]
    if len(ks) < 5: return None
    k0 = ks[len(ks) // 2]; pts = []
    for k in ks:
        s = shift(k0, k)
        if s is None: continue
        for c in cands[k]:
            if c[2] >= min_conf: pts.append((k, c[0] - s[0], c[1] - s[1], c[2]))
    if not pts: return 0.0, 0
    P = np.array([[p[1], p[2]] for p in pts]); best = (0.0, 0.0)
    for j in range(0, len(P), max(1, len(P) // 60)):
        near = np.linalg.norm(P - P[j], axis=1) < R
        fr = len({pts[i][0] for i in np.where(near)[0]}) / len(ks)
        if fr > best[0]: best = (fr, float(np.mean([pts[i][3] for i in np.where(near)[0]])))
    return round(best[0], 2), round(best[1], 2)
for i in range(30):
    if i in unsure: continue
    a0, a1, b0, b1 = ps[i]["_eps"]
    print(i, "FAKE" if i in fake else "real", stationary(a0, b1), "| window s", round((b1 - a0 + 1) / fps, 2))

print("--- fixed window -0.4..+0.8 s around the pass time, candidates in metres: best stationary share within R m")
from ipanema.calibration import to_m
def stat_m(t, R=1.5, min_conf=0.15, pre=0.4, post=0.8):
    ks = [k for k in range(int((t - pre) * fps), int((t + post) * fps) + 1) if k in cands and H.get(k) is not None]
    pts = []
    for k in ks:
        cc = [c for c in cands[k] if c[2] >= min_conf]
        if not cc: continue
        m = to_m(H[k], np.array([[c[0], c[1]] for c in cc], np.float32))
        for c, mm in zip(cc, m):
            if np.isfinite(mm).all() and -3 < mm[0] < L + 3 and -3 < mm[1] < W + 3: pts.append((k, mm[0], mm[1], c[2]))
    if not pts or not ks: return None
    P = np.array([[p[1], p[2]] for p in pts]); best = (0.0, 0.0)
    for j in range(len(P)):
        near = np.linalg.norm(P - P[j], axis=1) < R
        fr = len({pts[i][0] for i in np.where(near)[0]}) / len(ks)
        if fr > best[0]: best = (fr, float(np.mean([pts[i][3] for i in np.where(near)[0]])))
    return round(best[0], 2), round(best[1], 2), len(ks)
for i in range(30):
    if i in unsure: continue
    print(i, "FAKE" if i in fake else "real", stat_m(ps[i]["t"]), "| R=3:", stat_m(ps[i]["t"], R=3.0))

print("--- anchored: share of frames in [t-0.4, t+0.8] with a candidate (conf>=0.15) within R m of the ball pick at t; and the pick's spot")
def anchored(t, R=1.5, min_conf=0.15, pre=0.4, post=0.8):
    k0 = int(round(t * fps)); kk = [k for k in range(k0 - 3, k0 + 4) if k in ballm]
    if not kk: return None
    k0 = min(kk, key=lambda k: abs(k - k0)); p0 = np.asarray(ballm[k0])
    ks = [k for k in range(int((t - pre) * fps), int((t + post) * fps) + 1) if k in cands and H.get(k) is not None]
    hit = 0
    for k in ks:
        cc = [c for c in cands[k] if c[2] >= min_conf]
        if not cc: continue
        m = to_m(H[k], np.array([[c[0], c[1]] for c in cc], np.float32))
        if np.any(np.linalg.norm(m - p0, axis=1) < R): hit += 1
    return round(hit / max(1, len(ks)), 2), [round(float(x), 1) for x in p0]
for i in range(30):
    if i in unsure: continue
    print(i, "FAKE" if i in fake else "real", anchored(ps[i]["t"]), "| R=2.5:", anchored(ps[i]["t"], R=2.5)[0], "| post 1.5:", anchored(ps[i]["t"], post=1.5)[0])

print("--- plausible flight: longest run (s) of consecutive picked frames in [t-0.4,t+0.8] with step speed 2-35 m/s; metres covered in that run")
def flight(t, pre=0.4, post=0.8, lo=2.0, hi=35.0):
    ks = [k for k in range(int((t - pre) * fps), int((t + post) * fps) + 1) if k in ballm]
    best = (0, 0.0); run = []; 
    for a, b in zip(ks, ks[1:]):
        if b - a > 2: run = []; continue
        d = np.linalg.norm(np.asarray(ballm[b]) - np.asarray(ballm[a])); v = d * fps / (b - a)
        if lo <= v <= hi: run.append(d)
        else: run = []
        if run and (len(run), sum(run)) > best: best = (len(run), float(sum(run)))
    return round(best[0] / fps, 2), round(best[1], 1)
for i in range(30):
    if i in unsure: continue
    print(i, "FAKE" if i in fake else "real", flight(ps[i]["t"]), "| wider -1..+1.5:", flight(ps[i]["t"], 1.0, 1.5))
