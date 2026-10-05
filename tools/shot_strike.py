"""5 Oct (V1b): strike moment + origin of each Veo shot from the finder's ball track (results/kaggle/shot_ball_track) put
into metres with the exported frames' pitch_lines. Strike = the last moment the ball moves towards the attacked goal at
>= 8 m/s for 0.6 s, starting within 35 m of it. Writes results/review/vall/strikes.json for the by-eye check."""
import json, glob, bisect, numpy as np
M = "p15u-vs-vallentuna-2026-10-03-6cce"; L, W = 106.0, 64.0
T = json.load(open("results/kaggle/shot_ball_track/tracks.json"))["tracks"]
fr = []
for f in sorted(glob.glob(f"results/volume/runs/matches/{M}/frames_*.json")): fr += json.load(open(f))["frames"]
ts = [f["t"] for f in fr]
team = {int(s["t"]): s["team"] for s in json.load(open(f"results/volume/runs/matches/{M}/stats.json"))["metrics"]["shots"]}
def H_at(t):
    i = min(bisect.bisect_left(ts, t), len(ts) - 1)
    for j in (i, i - 1, i + 1):
        if 0 <= j < len(fr) and abs(fr[j]["t"] - t) < 0.2 and fr[j].get("pitch_lines"): return np.linalg.inv(np.array(fr[j]["pitch_lines"]).reshape(3, 3))
    return None
out = {}
for s, rows in T.items():
    s = int(s); G = np.array([L if team.get(s) == "A" else 0.0, W / 2]); pts = []
    for t, g in rows:
        if not g or g[0][2] < 0.3: continue
        Hi = H_at(t)
        if Hi is None: continue
        v = Hi @ np.array([g[0][0], g[0][1], 1.0]); m = v[:2] / v[2]
        if -5 < m[0] < L + 5 and -5 < m[1] < W + 5: pts.append((t, m, g[0][:2]))
    cand = []
    for i, (t, m, px) in enumerate(pts):
        later = [q for q in pts[i + 1:] if 0.4 <= q[0] - t <= 0.9]
        if not later: continue
        t2, m2, _ = later[0]; d1, d2 = np.linalg.norm(m - G), np.linalg.norm(m2 - G)
        if d1 <= 35 and (d1 - d2) / (t2 - t) >= 8 and d1 - d2 >= 4: cand.append((t, m.round(1).tolist(), px, round(float(d1), 1), round((d1 - d2) / (t2 - t), 1)))
    best = cand[-1] if cand else None
    out[s] = {"team": team.get(s), "points": len(pts), "candidates": cand[-4:], "strike": best}
    print(s, team.get(s), "points", len(pts), "strike", best[:2] + best[3:] if best else None, "| other cands", [c[0] for c in cand[-4:-1]])
json.dump(out, open("results/review/vall/strikes.json", "w"), indent=1)
