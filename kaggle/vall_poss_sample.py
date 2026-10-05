# Kaggle (free, CPU is enough): 60 random in-play Vallentuna frames (seed 7) at full resolution, for a by-eye
# "who has the ball" sample that measures possession independently -> /kaggle/working/poss/*.jpg + times.json
import subprocess, os, json, random
M = "p15u-vs-vallentuna-2026-10-03-6cce"; PER = [[435, 2890], [3220, 5645]]
subprocess.run("wget -q -O /kaggle/temp/v.mp4 " + "{{R2}}" + "/" + M + "/video.mp4", shell=True)
import cv2
random.seed(7); ts = sorted(round(random.uniform(*random.choice(PER)), 1) for _ in range(60))
os.makedirs("/kaggle/working/poss", exist_ok=True); cap = cv2.VideoCapture("/kaggle/temp/v.mp4"); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97; ok_t = []
for i, t in enumerate(ts):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read()
    if ok: cv2.imwrite(f"/kaggle/working/poss/p{i:02d}.jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85]); ok_t.append({"file": f"p{i:02d}.jpg", "t": t})
json.dump({"match": M, "fps": fps, "frames": ok_t}, open("/kaggle/working/poss/times.json", "w"), indent=1); print(len(ok_t), "frames")
