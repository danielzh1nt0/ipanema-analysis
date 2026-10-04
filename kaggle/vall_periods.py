# Kaggle (free, 4 Oct): Vallentuna kick-off and half-time by eye. Frames every 15 s: 0-8 min (sheet 0-1) and 44-58 min (sheet 2-4).
import cv2, numpy as np, json
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"; cap = cv2.VideoCapture(f"{R2}/p15u-vs-vallentuna-2026-10-03-6cce/video.mp4"); fps = cap.get(5)
def sheet(ts, name):
    tiles = []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, f = cap.read()
        if not ok: break
        g = cv2.resize(f, (640, 360)); cv2.putText(g, f"{t // 60}:{t % 60:02d} ({t}s)", (6, 24), 0, 0.8, (255, 255, 255), 2); tiles.append(g)
    while len(tiles) % 3: tiles.append(np.zeros((360, 640, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    for s in range(0, len(rows), 6): cv2.imwrite(f"{W}/{name}_{s // 6}.jpg", np.vstack(rows[s:s + 6]), [cv2.IMWRITE_JPEG_QUALITY, 80])
sheet(range(0, 480, 15), "start"); sheet(range(2640, 3480, 15), "half")
json.dump({"ok": True}, open(f"{W}/result.json", "w"))
