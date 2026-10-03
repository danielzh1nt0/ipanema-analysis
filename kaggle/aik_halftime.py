# Kaggle (free, 3 Oct): when does AIK's first half end? One frame every 15 s from 48:00 to 57:00, 1/3 size, time-stamped.
import cv2, numpy as np, json
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"; cap = cv2.VideoCapture(f"{R2}/p15u-vs-aik-2026-09-21-bd09/video.mp4"); fps = cap.get(5); tiles = []
for t in range(2880, 3420, 15):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, f = cap.read()
    if not ok: break
    g = cv2.resize(f, (640, 360)); cv2.putText(g, f"{t // 60}:{t % 60:02d} ({t}s)", (6, 24), 0, 0.8, (255, 255, 255), 2); tiles.append(g)
while len(tiles) % 3: tiles.append(np.zeros((360, 640, 3), np.uint8))
rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
for s in range(0, len(rows), 6): cv2.imwrite(f"{W}/sheet_{s // 6}.jpg", np.vstack(rows[s:s + 6]), [cv2.IMWRITE_JPEG_QUALITY, 80])
json.dump({"ok": True, "frames": len(tiles)}, open(f"{W}/result.json", "w"))
