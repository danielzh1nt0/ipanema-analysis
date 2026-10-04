# Kaggle (free, 4 Oct): Vallentuna (a) goal windows as strips of frames every 4 s (find the goal / kick-off by eye), the 2-1 gap
# every 12 s; (b) 40 random moments as full frames with a 100-px grid for the ball answer key (stage 1). Video from R2.
import json, cv2, urllib.request, numpy as np
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"
req = urllib.request.Request("https://raw.githubusercontent.com/danielzh1nt0/ipanema-analysis/main/results/review/vall/moments_fine.json", headers={"User-Agent": "Mozilla/5.0"})
D = json.load(urllib.request.urlopen(req)); cap = cv2.VideoCapture(f"{R2}/{D['src_key']}"); fps = cap.get(5)
def frame(t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read(); return cv2.resize(f, (1920, 1080)) if ok else np.zeros((1080, 1920, 3), np.uint8)
n = 0
for wi, w in enumerate(D["goal_windows"]):
    ts = list(np.arange(w["t0"], w["t1"], w["step"])); tiles = []
    for t in ts:
        g = cv2.resize(frame(t), (640, 360)); cv2.rectangle(g, (0, 0), (260, 18), (0, 0, 0), -1); cv2.putText(g, f"{w['label']} t={t:.0f}s ({int(t//60)}:{int(t%60):02d})", (3, 13), 0, 0.42, (255, 255, 255), 1); tiles.append(g)
    while len(tiles) % 3: tiles.append(np.zeros((360, 640, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    for s in range(0, len(rows), 4): cv2.imwrite(f"{W}/goal{wi}_{s // 4:02d}.jpg", np.vstack(rows[s:s + 4]), [cv2.IMWRITE_JPEG_QUALITY, 80]); n += 1
    print("window", wi, len(ts), flush=True)
for j, t in enumerate(D["ball_moments"]):
    g = frame(t)
    for x in range(0, 1920, 100): cv2.line(g, (x, 0), (x, 1080), (0, 255, 255) if x % 500 else (0, 0, 255), 1); cv2.putText(g, str(x), (x + 2, 14), 0, 0.4, (0, 255, 255), 1)
    for y in range(0, 1080, 100): cv2.line(g, (0, y), (1920, y), (0, 255, 255) if y % 500 else (0, 0, 255), 1); cv2.putText(g, str(y), (2, y - 2), 0, 0.4, (0, 255, 255), 1)
    cv2.putText(g, f"b{j:02d} t={t:.1f}s", (1700, 1070), 0, 0.6, (255, 255, 255), 2); cv2.imwrite(f"{W}/b{j:02d}.jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 88]); n += 1
json.dump({"ok": True, "files": n, "fps": fps}, open(f"{W}/result.json", "w")); print("done", n)
