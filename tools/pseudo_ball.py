"""K5 (16 Oct): ball pseudo-labels from a match's own export, no human in the loop. A frame is a label when the picked ball
was seen by the finder (not bridged) with confidence >= CONF, moved smoothly over the neighbouring exported frames, sat
within NEAR_M of a player, inside the pitch by EDGE_M, during play, and was not a ball lying still (>= 3 s). Labels are
at least GAP_S apart; frames within 1 s of a by-eye key (reference/<match>/ball_key_px.json) are left out so the keys
stay an honest test. -> results/ball/pseudo/<match>.json {video frame: [x, y]} at 1920x1080.
    python tools/pseudo_ball.py [match]"""
import sys, os, json, glob, math, collections
M = sys.argv[1] if len(sys.argv) > 1 else "p15u-vs-vallentuna-2026-10-03-6cce"
CONF, NEAR_M, EDGE_M, GAP_S, CAP = 0.8, 4.0, 1.5, 0.5, 3000
fs = []
for fn in sorted(glob.glob(f"results/volume/runs/matches/{M}/frames_*.json")): fs += json.load(open(fn))["frames"]
md = json.load(open(f"results/volume/runs/matches/{M}/match_data.json")); fps = md["fps"]; per = md["periods"]; L, W = md["pitch"]["length"], md["pitch"]["width"]
keys = [int(k) / fps for k in json.load(open(f"reference/{M}/ball_key_px.json"))] if os.path.exists(f"reference/{M}/ball_key_px.json") else []
inplay = lambda t: any(p["t_start"] + 2 <= t <= p["t_end"] - 2 for p in per)
def bm(f): b = f.get("ball"); return b["m"] if b and b.get("m") and b.get("state") == "observed" else None
out = {}; why = collections.Counter(); last = -1e9
for i, f in enumerate(fs):
    t = f["t"]; b = f.get("ball")
    if not inplay(t): why["out of play"] += 1; continue
    if not b or not b.get("px") or b.get("state") != "observed" or (b.get("conf") or 0) < CONF: why["ball not seen confidently"] += 1; continue
    x, y = b["m"]
    if not (EDGE_M <= x <= L - EDGE_M and EDGE_M <= y <= W - EDGE_M): why["near a line"] += 1; continue
    if not any(math.hypot(p["m"][0] - x, p["m"][1] - y) <= NEAR_M for p in f["players"]): why["nobody near"] += 1; continue
    # smooth motion over the neighbouring exported frames (10 a second)
    pm, nm = (bm(fs[i - 1]) if i else None), (bm(fs[i + 1]) if i + 1 < len(fs) else None)
    if pm is None or nm is None: why["neighbour frame has no seen ball"] += 1; continue
    dtp, dtn = t - fs[i - 1]["t"], fs[i + 1]["t"] - t
    if not (0.05 < dtp < 0.3 and 0.05 < dtn < 0.3): why["gap in frames"] += 1; continue
    v1 = ((x - pm[0]) / dtp, (y - pm[1]) / dtp); v2 = ((nm[0] - x) / dtn, (nm[1] - y) / dtn)
    sp = max(math.hypot(*v1), math.hypot(*v2))
    if sp > 30: why["too fast (jump)"] += 1; continue
    if math.hypot(v1[0] - v2[0], v1[1] - v2[1]) > 12 + 0.5 * sp: why["not smooth"] += 1; continue
    # a ball lying still for 3 s is a spare ball or a dead ball
    j = i
    while j > 0 and t - fs[j - 1]["t"] <= 3.0: j -= 1
    still = all(bm(g) is not None and math.hypot(bm(g)[0] - x, bm(g)[1] - y) < 0.5 for g in fs[j:i])
    if still and t - fs[j]["t"] >= 2.5: why["lying still"] += 1; continue
    if any(abs(t - k) <= 1.0 for k in keys): why["near a by-eye key (held out)"] += 1; continue
    if t - last < GAP_S: why["too close to the previous label"] += 1; continue
    out[int(round(t * fps))] = [round(b["px"][0], 1), round(b["px"][1], 1)]; last = t
    if len(out) >= CAP: break
os.makedirs("results/ball/pseudo", exist_ok=True)
json.dump(out, open(f"results/ball/pseudo/{M}.json", "w"))
print(len(out), "pseudo labels;", dict(why))
