# Kaggle (free, 5 Oct, V1b): ball track for each Veo shot window. Our RF-DETR ball finder on every 3rd frame (~10 fps) from
# +0 s to +30 s after each clip start; top 3 guesses per frame (full-res px + confidence). The strike is found locally from
# the track in metres (pitch_lines of the exported frames), then confirmed by eye on zoom crops.
import os, sys, json, subprocess, cv2, urllib.request, numpy as np
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"; T = "/kaggle/temp"
subprocess.run(f"git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {T}/ia", shell=True)
subprocess.run('pip install -q "rfdetr==1.11.0" supervision', shell=True); sys.path.insert(0, f"{T}/ia")
from ipanema import ballrf as BR
D = json.load(open(f"{T}/ia/results/review/vall/shot_origins.json")); cap = cv2.VideoCapture(f"{R2}/{D['src_key']}"); fps = cap.get(5)
ball = BR.load(f"{T}/ia/{BR.WEIGHTS}"); out = {}
for s in D["shots"]:
    k0, k1 = int(s * fps), int((s + 30) * fps); rows = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, k0); k = k0; batch = []
    while k < k1:
        ok, f = cap.read()
        if not ok: break
        if (k - k0) % 3 == 0: batch.append((k, cv2.resize(f, (1920, 1080))))
        if len(batch) == 4 or (k + 1 >= k1 and batch):
            res = BR.detect_many(ball, [b[1] for b in batch])
            for (kk, _), r in zip(batch, res): rows.append([round(kk / fps, 2), [[round(float(x), 1), round(float(y), 1), round(float(c), 3)] for x, y, c in r[:3]]])
            batch = []
        k += 1
    out[str(s)] = rows; print(s, len(rows), flush=True)
json.dump({"fps": fps, "tracks": out}, open(f"{W}/tracks.json", "w")); json.dump({"ok": True}, open(f"{W}/result.json", "w")); print("done")
