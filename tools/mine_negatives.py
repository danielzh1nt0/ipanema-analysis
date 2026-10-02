"""S8/B10 (2 Oct): 'not a ball' spots from the exact clip inputs, for retraining the ball finder with hard negatives.
Two sources, both facts: (1) at keyed moments (AIK 39, SFK-BP 34, B4 285) every candidate with conf >= MIN_CONF more than
FAR px from the keyed ball; (2) spots where a candidate (conf >= MIN_CONF) sits still (pan-corrected, within R px) for >= STILL_S
seconds with no player within NEAR_M for the whole spell (a dead ball has a player at it). Writes results/ball/negatives_2026-10-02.json
with {clip, frame, x, y, conf, why}. Free, local."""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL
MIN_CONF, FAR, R, STILL_S, NEAR_M = 0.3, 40.0, 25.0, 2.0, 3.0
STILL = False   # 2 Oct eye check (results/kaggle/neg_sheet): the still-nobody-near spots on AIK are mostly real balls at feet (players missing from the tracks) -> not used
def load(m):
    P_ = pickle.load(open(f"results/volume/cache/{m}/picker_inputs.pkl", "rb"))
    P_["per"] = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3])] for r in v] for k, v in P_["per"].items()}
    return P_
keys = {"p15u-vs-aik-2026-09-21-bd09_s2520": {int(k): v[:2] for k, v in json.load(open("reference/p15u-vs-aik-2026-09-21-bd09_s2520/ball_gt.json")).items()},
        "SFKBP1109_s1200": {int(k): v[:2] for k, v in json.load(open("results/volume/reference/SFKBP1109_s1200/ball_gt.json")).items() if v}}
for m in json.load(open("results/ball/b4/key.json"))["moments"]:
    if m["verdict"] == "ball": keys["SFKBP1109_s1200"].setdefault(m["frame"], (m["x"], m["y"]))
out = []
for clip, gt in keys.items():
    P_ = load(clip); cands, H, per, fps = P_["cands"], P_["H"], P_["per"], P_["fps"]
    n1 = 0
    for f, (gx, gy) in gt.items():
        for c in cands.get(f, []):
            if c[2] >= MIN_CONF and np.hypot(c[0] - gx, c[1] - gy) > FAR: out.append({"clip": clip, "frame": int(f), "x": float(c[0]), "y": float(c[1]), "conf": float(c[2]), "why": "keyed elsewhere", "ball": [float(gx), float(gy)]}); n1 += 1
    # still spots: follow each confident candidate forward
    n = len(H); used = set(); n2 = 0; step = 3
    for k0 in (range(0, n, step) if STILL else []):
        for c in cands.get(k0, []):
            if c[2] < MIN_CONF or (k0, round(c[0]), round(c[1])) in used: continue
            run = [(k0, c)]; last = k0; p = np.array([c[0], c[1]], np.float32); ok = True
            for k in range(k0 + 1, min(n, k0 + int(6 * fps))):
                q = BL._pan(H, last, k, [p])[0]
                near = [d for d in cands.get(k, []) if d[2] >= MIN_CONF and np.hypot(d[0] - q[0], d[1] - q[1]) <= R]
                if near:
                    d = max(near, key=lambda z: z[2]); run.append((k, d)); last = k; p = np.array([d[0], d[1]], np.float32)
                elif k - last > 0.5 * fps: break
            if (run[-1][0] - k0) / fps < STILL_S: continue
            # no player within NEAR_M for the whole spell (pitch metres of the spot via the frame's calibration)
            from ipanema.calibration import to_m
            far_from_all = True
            for k, d in run[::5]:
                try: mm = to_m(H[k], np.float32([[d[0], d[1]]]))[0]
                except Exception: continue
                if any(np.linalg.norm(r[2] - mm) < NEAR_M for r in per.get(k, [])): far_from_all = False; break
            if not far_from_all: continue
            for k, d in run[::int(fps // 3)]:
                out.append({"clip": clip, "frame": int(k), "x": float(d[0]), "y": float(d[1]), "conf": float(d[2]), "why": "still, nobody near"}); n2 += 1
            for k, d in run: used.add((k, round(d[0]), round(d[1])))
    print(clip, "keyed-elsewhere negatives", n1, "| still-spot negatives", n2)
json.dump(out, open("results/ball/negatives_2026-10-02.json", "w"))
print("total", len(out))
