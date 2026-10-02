# Kaggle (free, 2 Oct, S4): are our passes real? For each of the SFK-BP clip's 87 exported passes: 3 frames (t-0.4, t, t+0.8 s)
# with the ball pick (yellow ring), passer (green) and receiver (magenta) marked. Plus a dense sheet of minutes 1-3 (one frame
# per second, 1/4 size, ball ring) to count real passes by eye.
import json, cv2, numpy as np, urllib.request
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"
req = urllib.request.Request("https://raw.githubusercontent.com/danielzh1nt0/ipanema-analysis/main/results/review/passcheck_clip.json", headers={"User-Agent": "Mozilla/5.0"})
D = json.load(urllib.request.urlopen(req)); fps = D["fps"]; cap = cv2.VideoCapture(f"{R2}/SFKBP1109_s1200/video.mp4")
def frame(t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read()
    return cv2.resize(f, (1920, 1080)) if ok else np.zeros((1080, 1920, 3), np.uint8)
def key(t): return str(round(round(t * fps) / fps, 2))
def mark(f, t, p):
    b = D["ball"].get(key(t)); pl = D["players"].get(key(t), [])
    if b: cv2.circle(f, (int(b[0]), int(b[1])), 16, (0, 255, 255), 3)
    for pid, tm, px in pl:
        if pid == p["from"]: cv2.circle(f, (int(px[0]), int(px[1])), 12, (0, 255, 0), 3)
        if pid == p["to"]: cv2.circle(f, (int(px[0]), int(px[1])), 12, (255, 0, 255), 3)
    return b
tiles = []
for p in D["passes"]:
    row = []
    for d in (-0.4, 0.0, 0.8):
        t = p["t"] + d; f = frame(t); b = mark(f, t, p)
        c = b if b else (960, 540); x0 = int(min(max(c[0] - 320, 0), 1280)); y0 = int(min(max(c[1] - 180, 0), 720))
        tile = f[y0:y0 + 360, x0:x0 + 640].copy(); cv2.putText(tile, f"#{p['n']} t={p['t']:.1f} {d:+.1f}s {p['team']} {'ok' if p['completed'] else 'lost'}", (6, 24), 0, 0.7, (255, 255, 255), 2); row.append(tile)
    tiles.append(np.hstack(row))
for s in range(0, len(tiles), 6): cv2.imwrite(f"{W}/pass_{s // 6:02d}.jpg", np.vstack(tiles[s:s + 6]), [cv2.IMWRITE_JPEG_QUALITY, 80])
# dense sheet: minutes 1-3, one frame per second
small = []
for t in range(60, 180):
    f = frame(t); b = D["ball"].get(key(t))
    if b: cv2.circle(f, (int(b[0]), int(b[1])), 22, (0, 255, 255), 4)
    g = cv2.resize(f, (640, 360)); cv2.putText(g, f"{t}s", (6, 22), 0, 0.7, (255, 255, 255), 2); small.append(g)
for s in range(0, len(small), 12): cv2.imwrite(f"{W}/dense_{s // 12:02d}.jpg", np.vstack([np.hstack(small[i:i + 3]) for i in range(s, min(s + 12, len(small)), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
json.dump({"ok": True, "passes": len(tiles)}, open(f"{W}/result.json", "w"))
