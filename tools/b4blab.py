"""B4b (1 Oct): spare ball at the left goal post of SFK-BP. Picker v2 (RF-DETR + WASB 30, fused, new players) with and
without the recurring off-pitch spot rule (ball.recurring_spots), graded on the 34 moments and the B4 key (321).
Free, local. Writes results/ball/b4b/b4blab.json."""
import sys, os, json, gzip, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, linecal as LC
M = "SFKBP1109_s1200"; OUT = "results/ball/b4b"; os.makedirs(OUT, exist_ok=True)
d = np.load(f"results/picker/{M}.npz", allow_pickle=True); n = int(d["n"]); fps = float(d["fps"])
gt = {int(k): v for k, v in json.loads(bytes(d["gt"]).decode()).items() if v is not None}
rows = LC.find_rows(".", M); cal = LC.calibration_for_clip(rows[2], n, fps, 1920, 1080, offset_s=1200, log=lambda *a: None); H = cal["H"]; L, W = cal["L"], cal["W"]
def to_m(Hk, px): v = np.linalg.inv(Hk) @ np.array([px[0], px[1], 1.0]); return v[:2] / v[2]
R = json.load(gzip.open("results/qa/tracktest_gpu/rows_all.json.gz", "rt"))["rows"]["new, RF-DETR"]
per = {int(k): [[pid, t, to_m(H[int(k)], px), np.array(px), None, False] for pid, t, px, fl in v if px is not None] for k, v in R.items()}
o = pickle.load(open(f"results/volume/cache/{M}/ball_cands_wasb_1790008894_t2x2_thr0.05_lp30.pkl", "rb")); o = o[0] if isinstance(o, tuple) else o
w30 = {int(k): [(float(a), float(b), float(c)) for a, b, c in v] for k, v in o.items()}
rf = {}
for k, x, y, c in json.load(gzip.open(f"results/picker/ball_cands_rfdetr_{M}.json.gz", "rt")): rf.setdefault(int(k), []).append((x, y, c))
cd = BL.fuse_candidates(rf, w30)
key = json.load(open("results/ball/b4/key.json"))["moments"]
on = lambda b, f, x, y: f in b and np.hypot(b[f][0] - x, b[f][1] - y) <= 30
spare_m = np.median([to_m(H[m["frame"]], (m["x"], m["y"])) for m in key if m["verdict"] == "spare_ball"], axis=0)
def grade(b):
    balls = [m for m in key if m["verdict"] == "ball"]; restart = [m for m in balls if 240 <= m["id"] <= 252]
    spare = [m for m in key if m["verdict"] == "spare_ball"]
    # frames where the picker sits on the spare ball (within 2 m of its spot), whole clip
    on_spare = sum(1 for f, p in b.items() if np.hypot(*(to_m(H[f], p) - spare_m)) <= 2.0)
    return {k: (int(v) if isinstance(v, (np.integer, np.bool_)) else v) for k, v in {"moments34": sum(on(b, i, *g) for i, g in gt.items()), "b4_ball": sum(on(b, m["frame"], m["x"], m["y"]) for m in balls),
            "b4_ball_of": len(balls), "restart_kept": sum(on(b, m["frame"], m["x"], m["y"]) for m in restart), "restart_of": len(restart),
            "spare_still_picked": sum(on(b, m["frame"], m["x"], m["y"]) for m in spare), "spare_of": len(spare),
            "frames_on_spare_spot_s": round(on_spare / fps, 1), "frames_with_ball": len(b)}.items()}
res = {}
for r in (None, 1.5, 2.0, 2.5, 3.0):
    lg = []
    b = BL.bridge(BL.pick_v2(cd, H, L, W, per=per, fps=fps, recur_r=r, log=lg.append), fps)
    res["off" if r is None else f"recur_r {r} m"] = g = grade(b); g["log"] = lg
    print(r, {k: v for k, v in g.items()}, flush=True)
res["spare_spot_m"] = [round(float(v), 1) for v in spare_m]
json.dump(res, open(f"{OUT}/b4blab.json", "w"), indent=1)
# which known moments change with the chosen setting (2.0 m), and where the lost ones are
b0 = BL.bridge(BL.pick_v2(cd, H, L, W, per=per, fps=fps, log=lambda *a: None), fps)
b1 = BL.bridge(BL.pick_v2(cd, H, L, W, per=per, fps=fps, recur_r=2.0, log=lambda *a: None), fps)
ch = []
for i, g in gt.items():
    if on(b0, i, *g) != on(b1, i, *g): ch.append({"set": "34", "frame": i, "was_right": bool(on(b0, i, *g)), "ball_m": [round(float(v), 1) for v in to_m(H[i], g)]})
for m in key:
    if on(b0, m["frame"], m["x"], m["y"]) != on(b1, m["frame"], m["x"], m["y"]):
        ch.append({"set": "b4", "id": m["id"], "frame": m["frame"], "verdict": m["verdict"], "was_on": bool(on(b0, m["frame"], m["x"], m["y"])),
                   "guess_m": [round(float(v), 1) for v in to_m(H[m["frame"]], (m["x"], m["y"]))],
                   "new_pick_m": None if m["frame"] not in b1 else [round(float(v), 1) for v in to_m(H[m["frame"]], b1[m["frame"]])]})
res["changed_r2"] = ch
for c in ch: print(c)
json.dump(res, open(f"{OUT}/b4blab.json", "w"), indent=1)
# all stretches where old and new picks differ by > 30 px: one check moment every 0.5 s, for the picture sheet (free runner)
diff = [f for f in range(n) if (f in b0) != (f in b1) or (f in b0 and np.hypot(b0[f][0] - b1[f][0], b0[f][1] - b1[f][1]) > 30)]
segs = []
for f in diff:
    if segs and f - segs[-1][1] <= 3: segs[-1][1] = f
    else: segs.append([f, f])
chk = []
for a, z in segs:
    for f in range(a, z + 1, 15):
        k = len({c["frame"] for c in chk})
        for tag, b in (("old", b0), ("new", b1)):
            if f in b: chk.append({"id": k, "frame": f, "x": round(float(b[f][0]), 1), "y": round(float(b[f][1]), 1), "pick": tag})
print(f"{len(diff)} frames differ ({len(diff) / fps:.1f} s) in {len(segs)} stretches; {len(chk)} check tiles", flush=True)
res["differ"] = {"frames": len(diff), "seconds": round(len(diff) / fps, 1), "stretches": [[round(a / fps, 1), round(z / fps, 1)] for a, z in segs]}
json.dump(res, open(f"{OUT}/b4blab.json", "w"), indent=1)
json.dump({"clip": M, "moments": chk}, open(f"{OUT}/check_moments.json", "w"), indent=0)
