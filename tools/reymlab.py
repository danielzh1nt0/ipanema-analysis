"""E3 (28 Sep): who-has-the-ball on a second ground (Reymersholm, no pitch calibration), 5-min piece from 1500 s.
    python tools/reymlab.py strips   -> results/review/who_moments_reym.json (moments + players + ball) for tools/who_strips.py
    python tools/reymlab.py score    -> possession_simple (player-height ruler) vs answers in results/review/who_answers_reym.json
Inputs: ball guesses copied from Modal (results/volume/cache/<seg>/), player rows from the GPU run (results/qa/tracktest_gpu_<match>/)."""
import sys, os, json, gzip, glob, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, possession as P
MATCH = "p15u-vs-reymersholm-2026-09-18"; START = 1500; SEG = f"{MATCH}_s{START}_d300"
C = f"results/volume/cache/{SEG}/"
def load(f):
    o = pickle.load(open(f, "rb")); o = o[0] if isinstance(o, tuple) else o
    return {int(k): [(float(a), float(b), float(c)) for a, b, c in v] for k, v in o.items()}
clk = load(C + "ball_cands_clicks_st3_fz0.pkl"); wasb = load(sorted(glob.glob(C + "ball_cands_wasb_*_t2x2_thr0.05.pkl"))[-1])
R = json.load(gzip.open(f"results/qa/tracktest_gpu_{MATCH}/rows_all.json.gz", "rt")); fps = float(R["fps"])
rows = R["rows"]["new, RF-DETR"]; bh = R.get("box_h", {}).get("new, RF-DETR", {})
L, W = 106.0, 64.0; S_ = np.array([[1920 / L, 0, 0], [0, 1080 / W, 0], [0, 0, 1.0]]); n = max(int(k) for k in rows) + 1
n = max(n, max(clk) + 1, max(wasb) + 1); H = {k: S_ for k in range(n)}
per = {int(k): [[pid, t, np.array([px[0] * L / 1920, px[1] * W / 1080]), np.array(px), None, fl] for pid, t, px, fl in v if px is not None] for k, v in rows.items()}
boxh = {int(k): [h for (pid, t, px, fl), h in zip(rows[k], v) if px is not None] for k, v in bh.items()}
cands = BL.fuse_candidates(clk, wasb)
ball = BL.bridge(BL.pick_v2(cands, H, L, W, per=per, fps=fps, log=lambda *a: None), fps)
print(f"{n} frames, ball on {len(ball)}, guesses from clicks {len(clk)} frames / wasb {len(wasb)} frames, box heights {'yes' if boxh else 'NO'}")
if sys.argv[1] == "strips":
    mom = []; t = 3.0
    while t < 297 and len(mom) < 60: mom.append({"t": round(t, 2), "frame": int(round(t * fps)), "ours": None}); t += 4.9
    json.dump({"out": "results/qa/who_reym", "src_key": f"{MATCH}/video.mp4", "offset_frames": int(round(START * fps)), "moments": mom,
               "players": {str(m["frame"] + d): rows.get(str(m["frame"] + d), []) for m in mom for d in range(-24, 25, 6)},
               "ball": {str(m["frame"] + d): [round(v, 1) for v in ball[m["frame"] + d]] for m in mom for d in range(-24, 25, 6) if m["frame"] + d in ball}},
              open("results/review/who_moments_reym.json", "w"))
    print("moments", len(mom))
else:
    A = json.load(open("results/review/who_answers_reym.json"))["moments"]
    for near in (1.0, 1.5, 2.0):
        st = P.possession_simple(per, ball, H, n, near_m=near, boxh=boxh)
        r = {"team": [0, 0], "loose": [0, 0]}
        for a in A:
            if a["truth"] == "unsure": continue
            want = {"dark": 0, "white": 1, "light": 1, "loose": 2}[a["truth"]]; key = "loose" if want == 2 else "team"
            r[key][0] += int(st[a["frame"]] == want); r[key][1] += 1
        print(f"possession_simple near {near} m (player-height ruler): team {r['team'][0]}/{r['team'][1]}, loose {r['loose'][0]}/{r['loose'][1]}, all {r['team'][0] + r['loose'][0]}/{r['team'][1] + r['loose'][1]}")
