# Kaggle (free, 3 Oct, B6): grade ball finders on the Reymersholm eye-made key (results/review/b6/ball_gt.json, 26 moments chosen
# at random, no finder involved). For each finder: top guess within 30 px of the key, ball among the top 3, and on the 14
# unsure frames how often the finder fires with conf >= 0.3 (shouldn't much). Finders: B7 (in the app), hard-negatives retrain,
# at-feet retrain. Also WASB alone and the fused list (as the app picks from) when the WASB weights are on R2.
import os, sys, json, subprocess, urllib.request, cv2, numpy as np
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"; T = "/kaggle/temp"
subprocess.run(f"git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {T}/ia", shell=True); subprocess.run('pip install -q "rfdetr==1.11.0" supervision', shell=True)
sys.path.insert(0, f"{T}/ia"); from ipanema import ballrf as BR
D = json.load(open(f"{T}/ia/results/review/b6/moments.json")); K = {int(k): v for k, v in json.load(open(f"{T}/ia/results/review/b6/ball_gt.json")).items()}
cap = cv2.VideoCapture(f"{R2}/{D['src_key']}"); fps = cap.get(5); off = int(round(D["offset_s"] * fps)); frames = {}
for k in D["frames"]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, off + k); ok, f = cap.read()
    if ok: frames[k] = cv2.resize(f, (1920, 1080))
FINDERS = {"B7 (app)": BR.WEIGHTS, "hard negatives (2 Oct)": "results/kaggle/ballfinder_neg/rfdetr_ball_small_20261002_fp16.pth", "at feet (2 Oct)": "results/kaggle/ballfinder_feet/rfdetr_ball_small_20261002_fp16.pth"}
rep = {"moments": len(K), "unsure": len(frames) - len(K)}; rows = {}
near = lambda g, t: np.hypot(g[0] - t[0], g[1] - t[1]) <= 30
for name, w in FINDERS.items():
    try: m = BR.load(f"{T}/ia/{w}")
    except Exception as e: rep[name] = {"error": repr(e)[:200]}; continue
    ks = sorted(frames); out = BR.detect_many(m, [frames[k] for k in ks], floor=0.05); r = {"top": 0, "top3": 0, "any": 0, "unsure_fires": 0, "unsure_n": 0}; rows[name] = {}
    for k, g in zip(ks, out):
        rows[name][k] = [[round(x), round(y), round(c, 2)] for x, y, c in g[:5]]
        if k in K:
            t = K[k]; r["top"] += bool(g) and near(g[0], t); r["top3"] += any(near(z, t) for z in g[:3]); r["any"] += any(near(z, t) for z in g)
        else: r["unsure_n"] += 1; r["unsure_fires"] += bool(g) and g[0][2] >= 0.3
    rep[name] = r; print(name, r, flush=True); del m
json.dump({"summary": rep, "guesses": rows}, open(f"{W}/result.json", "w"), indent=1)
# sheet: key moments with the key (green) and each finder's top guess (B7 yellow, neg cyan, feet magenta)
cols = {"B7 (app)": (0, 255, 255), "hard negatives (2 Oct)": (255, 255, 0), "at feet (2 Oct)": (255, 0, 255)}; tiles = []
for k in sorted(K):
    t = K[k]; x0, y0 = int(min(max(0, t[0] - 160), 1600)), int(min(max(0, t[1] - 90), 900)); f = frames[k].copy()
    cv2.circle(f, (int(t[0]), int(t[1])), 14, (0, 255, 0), 2)
    for name, c in cols.items():
        g = rows.get(name, {}).get(k)
        if g: cv2.circle(f, (g[0][0], g[0][1]), 9, c, 2)
    tile = cv2.resize(f[y0:y0 + 180, x0:x0 + 320], (640, 360)); cv2.putText(tile, f"f{k}", (4, 18), 0, 0.6, (255, 255, 255), 2); tiles.append(tile)
while len(tiles) % 3: tiles.append(np.zeros((360, 640, 3), np.uint8))
for s in range(0, len(tiles), 9): cv2.imwrite(f"{W}/sheet_{s // 9}.jpg", np.vstack([np.hstack(tiles[i:i + 3]) for i in range(s, min(len(tiles), s + 9), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("done")
