# Kaggle (free, 30 Sep/1 Oct): blind ball key for the AIK app clip. 60 moments (every 5 s). For each, zoomed crops at the
# new finder's top-2 guesses, shown as A/B in random order; Claude grades by eye which one is the ball before opening key.json.
import json, gzip, random, urllib.request, cv2, numpy as np
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"; M = "p15u-vs-aik-2026-09-21-bd09_s2520"
def fetch(u): return urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=300).read()
rows = json.loads(gzip.decompress(fetch(f"https://raw.githubusercontent.com/danielzh1nt0/ipanema-analysis/main/results/picker/ball_cands_rfdetr_{M}.json.gz")))
G = {}
for k, x, y, c in rows: G.setdefault(int(k), []).append((x, y, c))
cap = cv2.VideoCapture(f"{R2}/{M}/video.mp4"); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
rnd = random.Random(11); key = {}; made = 0
def crop(f, p):
    x0 = int(min(max(p[0] - 160, 0), 1600)); y0 = int(min(max(p[1] - 90, 0), 900))
    c = cv2.resize(f[y0:y0 + 180, x0:x0 + 320], (960, 540), interpolation=cv2.INTER_CUBIC); cv2.drawMarker(c, (int((p[0] - x0) * 3), int((p[1] - y0) * 3)), (255, 0, 255), cv2.MARKER_CROSS, 16, 1); return c
for j in range(60):
    k = int(round((2.5 + 5 * j) * fps)); g = G.get(k) or G.get(k + 1) or G.get(k - 1)
    if not g: key[j] = {"frame": k, "skip": "no guesses"}; continue
    a = g[0]; b = next((q for q in g[1:] if np.hypot(q[0] - a[0], q[1] - a[1]) > 40), None)
    cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
    if not ok: key[j] = {"frame": k, "skip": "no frame"}; continue
    f = cv2.resize(f, (1920, 1080)); pair = [("top1", a)] + ([("top2", b)] if b else [])
    rnd.shuffle(pair); tiles = [crop(f, p) for _, p in pair]
    if len(tiles) == 1: tiles.append(np.zeros_like(tiles[0]))
    im = np.hstack([tiles[0], np.full((540, 8, 3), 255, np.uint8), tiles[1]])
    cv2.putText(im, f"m{j:02d}  LEFT=A  RIGHT=B", (10, 30), 0, 0.9, (255, 255, 255), 2)
    small = cv2.resize(f, (960, 540)); cv2.putText(small, "context (no marks)", (10, 30), 0, 0.9, (255, 255, 255), 2)
    cv2.imwrite(f"{W}/ab{j:02d}.jpg", np.vstack([im, np.hstack([small, np.zeros((540, 968, 3), np.uint8)])]), [cv2.IMWRITE_JPEG_QUALITY, 85]); made += 1
    key[j] = {"frame": k, "A": [pair[0][0], [round(v, 1) for v in pair[0][1]]], "B": [pair[1][0], [round(v, 1) for v in pair[1][1]]] if len(pair) > 1 else None}
json.dump(key, open(f"{W}/key_HIDDEN.json", "w")); json.dump({"ok": True, "pictures": made, "fps": fps, "frames": n}, open(f"{W}/result.json", "w"))
