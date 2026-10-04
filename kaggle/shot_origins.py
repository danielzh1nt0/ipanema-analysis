# Kaggle (free, 5 Oct, V1): find where each Veo shot was struck. Per shot: frames every 1 s from +3 s to +27 s after the clip
# start, 960x540 each with a 100-px (full-res) grid, 6 per sheet. Video from R2.
import json, cv2, urllib.request, numpy as np
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"
req = urllib.request.Request("https://raw.githubusercontent.com/danielzh1nt0/ipanema-analysis/main/results/review/vall/shot_origins.json", headers={"User-Agent": "Mozilla/5.0"})
D = json.load(urllib.request.urlopen(req)); cap = cv2.VideoCapture(f"{R2}/{D['src_key']}"); fps = cap.get(5); n = 0
for s in D["shots"]:
    tiles = []
    for t in np.arange(s + D["t_from"], s + D["t_to"] + 0.01, D["step"]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read()
        f = cv2.resize(f, (1920, 1080)) if ok else np.zeros((1080, 1920, 3), np.uint8)
        for x in range(0, 1920, 100): cv2.line(f, (x, 0), (x, 1080), (0, 255, 255) if x % 500 else (0, 0, 255), 1)
        for y in range(0, 1080, 100): cv2.line(f, (0, y), (1920, y), (0, 255, 255) if y % 500 else (0, 0, 255), 1)
        g = cv2.resize(f, (960, 540)); cv2.rectangle(g, (0, 0), (300, 24), (0, 0, 0), -1); cv2.putText(g, f"shot {s}  t={t:.0f}s (+{t - s:.0f})", (4, 17), 0, 0.55, (255, 255, 255), 1); tiles.append(g)
    while len(tiles) % 6: tiles.append(np.zeros((540, 960, 3), np.uint8))
    for i in range(0, len(tiles), 6):
        cv2.imwrite(f"{W}/s{s}_{i // 6}.jpg", np.vstack([np.hstack(tiles[i + j:i + j + 2]) for j in (0, 2, 4)]), [cv2.IMWRITE_JPEG_QUALITY, 80]); n += 1
    print(s, flush=True)
json.dump({"ok": True, "sheets": n, "fps": fps}, open(f"{W}/result.json", "w")); print("done", n)
