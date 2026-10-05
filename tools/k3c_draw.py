"""K3c: draw an override's teams on the 12 Kaggle QA frames (red dot SFK, blue Vallentuna) + owner check.
    python tools/k3c_draw.py <override.json> <out_dir>"""
import sys, os, json, glob, bisect, cv2, collections, math
M = "p15u-vs-vallentuna-2026-10-03-6cce"; D = "results/kaggle/vall_relabel/relabel"
ov = json.load(open(sys.argv[1])); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
def team(p, t):
    v = ov.get(str(p["id"]), p["team"])
    if isinstance(v, list): v = next((tt for a, b, tt in v if a <= t <= b), p["team"])
    return v
frames = []
for fn in sorted(glob.glob(f"results/volume/runs/matches/{M}/frames_*.json")): frames += json.load(open(fn))["frames"]
ts = [f["t"] for f in frames]; c = collections.Counter(); n = 0
for fr in frames:
    if fr["players"]:
        n += 1
        for p in fr["players"]: c[team(p, fr["t"])] += 1
print("per frame", {k: round(v / n, 1) for k, v in c.items()})
for t in [638, 1045, 1452, 1858, 2265, 2672, 3408, 3815, 4222, 4628, 5035, 5442]:
    i = min(bisect.bisect_left(ts, t), len(ts) - 1); fr = frames[i]; f = cv2.imread(f"{D}/qa/t{t}.jpg"); cnt = collections.Counter()
    for p in fr["players"]:
        tm = team(p, fr["t"]); cnt[tm] += 1; x, y = map(int, p["px"]); cv2.circle(f, (x, y + 8), 11, (0, 0, 255) if tm == "A" else (255, 120, 0), -1)
    cv2.putText(f, f"t={t} SFK(red dot)={cnt['A']} VAL(blue dot)={cnt['B']}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    cv2.imwrite(f"{out}/t{t}.jpg", cv2.resize(f, (1280, 720)))
key = json.load(open(f"reference/{M}/ball_key_graded.json"))["items"]; r = collections.Counter()
for it in key:
    if it.get("owner") not in ("A", "B") or not it.get("ball_px"): continue
    i = min(bisect.bisect_left(ts, it["t"]), len(ts) - 1); fr = frames[i]; gx, gy = it["ball_px"]
    ps = sorted(fr["players"], key=lambda p: math.hypot(p["px"][0] - gx, p["px"][1] - gy))
    if ps: r["right" if team(ps[0], fr["t"]) == it["owner"] else "wrong"] += 1
print("owner nearest", dict(r))
