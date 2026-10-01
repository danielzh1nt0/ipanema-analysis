"""B4 (1 Oct): pick candidate ball-key moments on the SFK-BP clip where the app picker (RF-DETR + WASB 30 fused, v2) and
both finders' top guesses agree. Writes results/ball/b4/candidates.json and checks the rule on the 34 known moments.
Free, local. The picture sheets are cut on the free runner (tools/b4_sheet.py), then graded by eye."""
import sys, os, json, gzip, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, linecal as LC, ballkey_auto as AK
M = "SFKBP1109_s1200"; OUT = "results/ball/b4"; os.makedirs(OUT, exist_ok=True)
d = np.load(f"results/picker/{M}.npz", allow_pickle=True); n = int(d["n"]); fps = float(d["fps"])
gt = {int(k): v for k, v in json.loads(bytes(d["gt"]).decode()).items() if v is not None}
gt_all = [int(k) for k in json.loads(bytes(d["gt"]).decode())]
rows = LC.find_rows(".", M); cal = LC.calibration_for_clip(rows[2], n, fps, 1920, 1080, offset_s=1200, log=lambda *a: None); H = cal["H"]; L, W = cal["L"], cal["W"]
def to_m(Hk, px): v = np.linalg.inv(Hk) @ np.array([px[0], px[1], 1.0]); return v[:2] / v[2]
R = json.load(gzip.open("results/qa/tracktest_gpu/rows_all.json.gz", "rt"))["rows"]["new, RF-DETR"]
per = {int(k): [[pid, t, to_m(H[int(k)], px), np.array(px), None, False] for pid, t, px, fl in v if px is not None] for k, v in R.items()}
o = pickle.load(open(f"results/volume/cache/{M}/ball_cands_wasb_1790008894_t2x2_thr0.05_lp30.pkl", "rb")); o = o[0] if isinstance(o, tuple) else o
w30 = {int(k): [(float(a), float(b), float(c)) for a, b, c in v] for k, v in o.items()}
rf = {}
for k, x, y, c in json.load(gzip.open(f"results/picker/ball_cands_rfdetr_{M}.json.gz", "rt")): rf.setdefault(int(k), []).append((x, y, c))
pick = BL.pick_v2(BL.fuse_candidates(rf, w30), H, L, W, per=per, fps=fps, log=print)   # raw path, no bridged frames
# the rule on the 34 known moments (no spacing, no steadiness): how often it fires, and how often it is right
fire = [i for i in gt if AK.agreed(i, pick, rf, w30)]
right = [i for i in fire if np.hypot(pick[i][0] - gt[i][0], pick[i][1] - gt[i][1]) <= 30]
sel = AK.select(pick, rf, w30, n, gap=15, steady=0, avoid=gt_all)   # steady=1 halves it (221); the eye check is the real filter
print(f"rule fires on {len(fire)}/{len(gt)} known moments, right on {len(right)}; {len(sel)} candidates selected", flush=True)
json.dump({"clip": M, "fps": fps, "rule": "picker + RF-DETR top (>=0.4) + WASB top (>=0.5) within 10 px, >= 15 frames (0.5 s) apart, not next to a known moment",
           "known_check": {"fires": len(fire), "of": len(gt), "right": len(right)},
           "moments": [{"id": k, "frame": f, "x": round(x, 1), "y": round(y, 1), "rf": round(a, 3), "wasb": round(b, 3)} for k, (f, x, y, a, b) in enumerate(sel)]},
          open(f"{OUT}/candidates.json", "w"), indent=0)
