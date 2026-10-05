"""5 Oct: who has the ball? The exported Vallentuna frames vs the by-eye owners in ball_key_graded.json (owner A/B).
For each owned moment: the team of the player nearest the true ball (by feet pixel) in the export, and the team the
export's own possession field says (if present). Needs the fetched frame chunks in results/volume.
    PYTHONPATH=. python tools/vall_owner_check.py"""
import json, glob, bisect, math, collections
M = "p15u-vs-vallentuna-2026-10-03-6cce"; D = f"results/volume/runs/matches/{M}"
key = json.load(open(f"reference/{M}/ball_key_graded.json"))["items"]
frames = []
for f in sorted(glob.glob(f"{D}/frames_*.json")): frames += json.load(open(f))["frames"]
ts = [f["t"] for f in frames]
res = collections.Counter()
for it in key:
    if it.get("owner") not in ("A", "B") or not it.get("ball_px"): continue
    i = min(bisect.bisect_left(ts, it["t"]), len(ts) - 1); fr = frames[i]
    if abs(fr["t"] - it["t"]) > 0.3: res["no frame"] += 1; print(it["id"], "no frame"); continue
    gx, gy = it["ball_px"]
    ps = sorted(fr.get("players", []), key=lambda p: math.hypot(p["px"][0] - gx, p["px"][1] - gy))
    near = ps[0]["team"] if ps else None; d = round(math.hypot(ps[0]["px"][0] - gx, ps[0]["px"][1] - gy)) if ps else None
    poss = fr.get("possession") or (fr.get("ball") or {}).get("team")
    ok = near == it["owner"]; res["nearest right" if ok else "nearest wrong"] += 1
    if poss is not None: res["poss right" if poss == it["owner"] else "poss wrong"] += 1
    print(it["id"], it["t"], "truth", it["owner"], "nearest", near, f"{d}px", "export-possession", poss, "OK" if ok else "WRONG")
print(dict(res))
