# Kaggle (free): two 60-s windows of Vallentuna at 3 pictures per second, as numbered contact sheets (12 per sheet),
# for counting passes by eye -> /kaggle/working/pc/w<k>_s<nn>.jpg
import subprocess, os, json
M = "p15u-vs-vallentuna-2026-10-03-6cce"; WIN = [4000, 1200]; FPS = 2
os.makedirs("/kaggle/temp", exist_ok=True); subprocess.run("wget -q -O /kaggle/temp/v.mp4 " + "{{R2}}" + "/" + M + "/video.mp4", shell=True)
import cv2, numpy as np
os.makedirs("/kaggle/working/pc", exist_ok=True); cap = cv2.VideoCapture("/kaggle/temp/v.mp4"); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
for w, t0 in enumerate(WIN):
    tiles = []
    for j in range(60 * FPS):
        t = t0 + j / FPS; cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read()
        if not ok: continue
        f = cv2.resize(f, (1920, 1080))[230:850, 410:1510]; f = cv2.resize(f, (1000, 564))
        cv2.putText(f, f"{t:.1f}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2); tiles.append(f)
    for s in range(0, len(tiles), 6):
        q = tiles[s:s + 6]
        while len(q) < 6: q.append(np.zeros_like(tiles[0]))
        cv2.imwrite(f"/kaggle/working/pc/w{w}_s{s // 6:02d}.jpg", np.vstack([np.hstack(q[r:r + 2]) for r in range(0, 6, 2)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("done")
