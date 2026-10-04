"""4 Oct: the app's exported ball vs the Vallentuna ball key (reference/<match>/ball_gt.json, 29 clear moments). For each
moment: the exported ball pixel at that time, distance to the true ball, and where the export put it instead (near a
spare-ball spot by the fence/goal, on a player, or nothing). Needs the fetched frame chunks.
    PYTHONPATH=. python tools/vall_ball_check.py"""
import json, glob, bisect, math, collections
M = "p15u-vs-vallentuna-2026-10-03-6cce"; D = f"results/volume/runs/matches/{M}"
key = json.load(open(f"reference/{M}/ball_key_graded.json"))["items"]; G = json.load(open("results/kaggle/vall_ball_key/guesses.json"))
frames = []
for f in sorted(glob.glob(f"{D}/frames_*.json")): frames += json.load(open(f))["frames"]
ts = [f["t"] for f in frames]; print("frames", len(frames))
res = collections.Counter(); rows = []
for it in key:
    if it["grade"] != "A": continue
    i = min(bisect.bisect_left(ts, it["t"]), len(ts) - 1); fr = frames[i]
    if abs(fr["t"] - it["t"]) > 0.2: res["no frame"] += 1; rows.append((it["id"], "no frame")); continue
    b = fr.get("ball"); gx, gy = it["ball_px"]
    if not b: res["no ball"] += 1; rows.append((it["id"], "no ball", fr.get("cal_ok"))); continue
    d = math.hypot(b["px"][0] - gx, b["px"][1] - gy)
    # is the export on another finder guess (B/C) = a spare ball the finder also saw?
    other = [k for k, (x, y, c) in zip("ABC", G[it["id"]]["guesses"]) if k != "A" and math.hypot(b["px"][0] - x, b["px"][1] - y) < 30]
    kind = "right" if d < 30 else ("other guess " + "".join(other) if other else "elsewhere")
    res[kind.split(" ")[0] if kind.startswith("other") else kind] += 1; rows.append((it["id"], kind, round(d), b.get("state"), fr.get("cal_ok")))
for r in rows: print(*r)
print(dict(res))
