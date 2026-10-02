# Kaggle (free, 3 Oct): three full-size frames of bundesliga1 (Drive, link-shared) at 10 s, 100 s, 190 s, plus a 2x zoom of
# the pitch area, so pitch points can be read off by eye for a fixed-camera calibration. No model, CPU only.
import os, subprocess, cv2, json
W = "/kaggle/working"; T = "/kaggle/temp"; os.makedirs(T, exist_ok=True)
subprocess.run("pip install -q gdown", shell=True); dst = f"{T}/b1.mp4"
subprocess.run(["gdown", "--fuzzy", "https://drive.google.com/file/d/1aQs8FTVbZm7ewmDzvaq658d1CdV5k9dR/view", "-O", dst], capture_output=True, text=True)
cap = cv2.VideoCapture(dst); fps = cap.get(5); info = {"fps": fps, "frames": int(cap.get(7)), "size": [int(cap.get(3)), int(cap.get(4))]}
for t in (10, 100, 190):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, f = cap.read()
    if not ok: continue
    cv2.imwrite(f"{W}/frame_{t:03d}.png", f)
    # grid every 100 px to read coordinates off the picture
    g = f.copy()
    for x in range(0, g.shape[1], 100): cv2.line(g, (x, 0), (x, g.shape[0]), (0, 255, 255) if x % 500 else (0, 0, 255), 1); cv2.putText(g, str(x), (x + 2, 14), 0, 0.4, (0, 255, 255), 1)
    for y in range(0, g.shape[0], 100): cv2.line(g, (0, y), (g.shape[1], y), (0, 255, 255) if y % 500 else (0, 0, 255), 1); cv2.putText(g, str(y), (2, y - 2), 0, 0.4, (0, 255, 255), 1)
    cv2.imwrite(f"{W}/grid_{t:03d}.jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 92])
    for name, (x0, y0, x1, y1) in {"left": (0, 150, 960, 650), "right": (960, 150, 1920, 650)}.items():
        z = cv2.resize(g[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC); cv2.imwrite(f"{W}/zoom_{t:03d}_{name}.jpg", z, [cv2.IMWRITE_JPEG_QUALITY, 92])
json.dump(info, open(f"{W}/result.json", "w")); print(info)
