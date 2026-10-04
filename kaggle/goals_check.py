# Kaggle (free, 4 Oct): who scored? Frames every 2 s around the AIK first-half goals (46:02, 52:08) and the SFK-BP first-half goal
# (43:39), from 22 s before to 24 s after the Veo clip time. Video from R2.
import json, cv2, urllib.request, numpy as np
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"
req = urllib.request.Request("https://raw.githubusercontent.com/danielzh1nt0/ipanema-analysis/main/results/review/vall/goals_check.json", headers={"User-Agent": "Mozilla/5.0"})
D = json.load(urllib.request.urlopen(req)); n = 0
for wi, w in enumerate(D["goal_windows"]):
    cap = cv2.VideoCapture(f"{R2}/{D['srcs'][w['label']]}"); fps = cap.get(5); tiles = []
    for t in np.arange(w["t0"], w["t1"], w["step"]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read()
        g = cv2.resize(f, (640, 360)) if ok else np.zeros((360, 640, 3), np.uint8)
        cv2.rectangle(g, (0, 0), (250, 18), (0, 0, 0), -1); cv2.putText(g, f"{w['label']} t={t:.0f}s", (3, 13), 0, 0.42, (255, 255, 255), 1); tiles.append(g)
    while len(tiles) % 3: tiles.append(np.zeros((360, 640, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    for s in range(0, len(rows), 4): cv2.imwrite(f"{W}/g{wi}_{s // 4}.jpg", np.vstack(rows[s:s + 4]), [cv2.IMWRITE_JPEG_QUALITY, 82]); n += 1
json.dump({"ok": True, "files": n}, open(f"{W}/result.json", "w")); print("done", n)
