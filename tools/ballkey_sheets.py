"""29 Sep: picture sheets for a bigger ball answer key (150 moments: 90 SFK-BP clip, 60 Reymersholm night), graded by
Claude by eye: for each moment the frame (wide, guesses numbered) + 2x zoom crops of each numbered guess over 3 frames
(-4, 0, +4) so a moving ball shows. Answer = the number that is the match ball, or 0 = none / unsure. Free runner (R2).
-> results/qa/ballkey/sheet_*.jpg"""
import os, json, cv2, numpy as np
D = json.load(open("results/review/ballkey_moments.json")); OUT = "results/qa/ballkey"; os.makedirs(OUT, exist_ok=True)
base = os.environ.get("LOCAL_CLIP_DIR"); R2 = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
caps = {}
def frame(m, d):
    src = m["src_key"]
    if src not in caps: caps[src] = cv2.VideoCapture(f"{base}/fake.mp4" if base else f"{R2}/{src}")
    c = caps[src]; c.set(cv2.CAP_PROP_POS_FRAMES, m["offset"] + m["frame"] + d); ok, f = c.read()
    return f if ok else np.zeros((1080, 1920, 3), np.uint8)
tiles = []
for i, m in enumerate(D["moments"]):
    f0 = frame(m, 0); fs = {d: frame(m, d) for d in (-4, 4)}
    wide = f0.copy()
    for j, (x, y, s, src) in enumerate(m["guesses"], 1):
        col = (0, 255, 255) if src == "w" else (255, 0, 255)
        cv2.circle(wide, (int(x), int(y)), 16, col, 2); cv2.putText(wide, str(j), (int(x) + 14, int(y) - 12), 0, 1.1, (0, 0, 0), 5); cv2.putText(wide, str(j), (int(x) + 14, int(y) - 12), 0, 1.1, col, 2)
    wide = cv2.resize(wide, (960, 540)); cv2.putText(wide, f"#{i} {m['set'][:18]} f{m['frame']}", (8, 30), 0, 0.9, (255, 255, 255), 2)
    crops = []
    for j, (x, y, s, src) in enumerate(m["guesses"], 1):
        row = []
        for d, f in ((-4, fs[-4]), (0, f0), (4, fs[4])):
            x0 = int(min(max(x - 40, 0), 1920 - 80)); y0 = int(min(max(y - 40, 0), 1080 - 80))
            cr = cv2.resize(f[y0:y0 + 80, x0:x0 + 80], (120, 120), interpolation=cv2.INTER_CUBIC)
            if d == 0: cv2.circle(cr, (int((x - x0) * 1.5), int((y - y0) * 1.5)), 14, (0, 255, 255), 1)
            row.append(cr)
        r = np.hstack(row); cv2.putText(r, str(j), (4, 26), 0, 0.9, (0, 0, 0), 4); cv2.putText(r, str(j), (4, 26), 0, 0.9, (255, 255, 255), 2); crops.append(r)
    while len(crops) < 10: crops.append(np.zeros((120, 360, 3), np.uint8))
    grid = np.vstack([np.hstack(crops[k:k + 2]) for k in range(0, 10, 2)])        # 5 rows x 2 = 600 x 720
    grid = cv2.resize(grid, (960, 800))
    tiles.append(np.vstack([wide, grid]))
for s in range(0, len(tiles), 2):
    pair = tiles[s:s + 2]
    if len(pair) < 2: pair.append(np.zeros_like(pair[0]))
    cv2.imwrite(f"{OUT}/sheet_{s // 2:03d}.jpg", np.hstack(pair), [cv2.IMWRITE_JPEG_QUALITY, 82])
print("sheets", (len(tiles) + 1) // 2)
