# Kaggle (free, 3 Oct, B6): the 40 random Reymersholm moments as full frames with a 100-px grid (stage 1: find the ball by eye),
# read from R2 (match video from 1500 s). No finder, no markers.
import json, cv2, urllib.request, os
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"
req = urllib.request.Request("https://raw.githubusercontent.com/danielzh1nt0/ipanema-analysis/main/results/review/b6/moments.json", headers={"User-Agent": "Mozilla/5.0"})
D = json.load(urllib.request.urlopen(req)); cap = cv2.VideoCapture(f"{R2}/{D['src_key']}"); fps = cap.get(5); off = int(round(D["offset_s"] * fps)); n = 0
for j, k in enumerate(D["frames"]):
    cap.set(cv2.CAP_PROP_POS_FRAMES, off + k); ok, f = cap.read()
    if not ok: continue
    f = cv2.resize(f, (1920, 1080)); g = f.copy()
    for x in range(0, 1920, 100): cv2.line(g, (x, 0), (x, 1080), (0, 255, 255) if x % 500 else (0, 0, 255), 1); cv2.putText(g, str(x), (x + 2, 14), 0, 0.4, (0, 255, 255), 1)
    for y in range(0, 1080, 100): cv2.line(g, (0, y), (1920, y), (0, 255, 255) if y % 500 else (0, 0, 255), 1); cv2.putText(g, str(y), (2, y - 2), 0, 0.4, (0, 255, 255), 1)
    cv2.putText(g, f"m{j:02d} frame {k}", (1700, 1070), 0, 0.6, (255, 255, 255), 2)
    cv2.imwrite(f"{W}/m{j:02d}.jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 90]); n += 1
json.dump({"ok": True, "frames": n, "fps": fps}, open(f"{W}/result.json", "w")); print("done", n)
