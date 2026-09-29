# Kaggle (free, 29 Sep): 4 full Reymersholm frames where the new ball finder and the old ball disagree (possession flipped).
# Marks: yellow circle = old ball, red cross = new ball's pick. Plus the new finder's picks 1 s before/after (drift = moving).
import os, json, cv2, numpy as np, urllib.request
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"; OFF = 44955
PTS = {5230: ((1401, 623), (870, 511)), 6845: ((1498, 508), (352, 587)), 7579: ((409, 435), (754, 528)), 8754: ((1383, 597), (858, 449))}
cap = cv2.VideoCapture(f"{R2}/p15u-vs-reymersholm-2026-09-18/video.mp4"); tiles = []
for k, (o, n) in PTS.items():
    for d in (-30, 0, 30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, OFF + k + d); ok, f = cap.read()
        if not ok: continue
        f = cv2.resize(f, (1920, 1080))
        if d == 0:
            cv2.circle(f, o, 22, (0, 255, 255), 3); cv2.drawMarker(f, n, (0, 0, 255), cv2.MARKER_CROSS, 40, 4)
            cv2.imwrite(f"{W}/full_{k}.jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
        for p, c in ((o, (0, 255, 255)), (n, (0, 0, 255))):
            x0, y0 = int(min(max(p[0] - 150, 0), 1620)), int(min(max(p[1] - 100, 0), 880)); cr = f[y0:y0 + 200, x0:x0 + 300].copy()
            cv2.putText(cr, f"{k}{d:+d} {'old' if c[0] == 0 else 'new'}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1); tiles.append(cr)
cv2.imwrite(f"{W}/sheet.jpg", np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles) - len(tiles) % 6, 6)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
json.dump({"ok": True, "tiles": len(tiles)}, open(f"{W}/result.json", "w"))
