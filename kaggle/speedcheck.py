# Kaggle (free, 1 Oct): by-eye check of player speeds. For 28 readings (fast far-side, fast near-side, normal; shuffled,
# speed NOT printed on the picture), a crop at t and t+0.5 s around the player, with a cross at both positions.
import json, cv2, numpy as np, urllib.request
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"
req = urllib.request.Request("https://raw.githubusercontent.com/danielzh1nt0/ipanema-analysis/main/results/review/speedcheck_moments.json", headers={"User-Agent": "Mozilla/5.0"})
M = json.load(urllib.request.urlopen(req)); cap = cv2.VideoCapture(f"{R2}/SFKBP1109_s1200/video.mp4"); tiles = []
for m in M:
    cx = int((m["px"][0] + m["px2"][0]) / 2); cy = int((m["px"][1] + m["px2"][1]) / 2)
    x0 = min(max(cx - 240, 0), 1440); y0 = min(max(cy - 160, 0), 760); row = []
    for d, p in ((0, m["px"]), (15, m["px2"])):
        cap.set(cv2.CAP_PROP_POS_FRAMES, m["frame"] + d); ok, f = cap.read()
        f = cv2.resize(f, (1920, 1080)) if ok else np.zeros((1080, 1920, 3), np.uint8)
        cv2.drawMarker(f, (int(p[0]), int(p[1])), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
        t = f[y0:y0 + 320, x0:x0 + 480].copy(); cv2.putText(t, f"#{m['n']} {'+0.5s' if d else '0s'}", (6, 22), 0, 0.7, (255, 255, 255), 2); row.append(t)
    tiles.append(np.hstack(row))
for s in range(0, len(tiles), 4): cv2.imwrite(f"{W}/sheet_{s // 4}.jpg", np.vstack(tiles[s:s + 4]), [cv2.IMWRITE_JPEG_QUALITY, 85])
json.dump({"ok": True, "n": len(tiles)}, open(f"{W}/result.json", "w"))
