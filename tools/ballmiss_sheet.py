"""B5 (28 Sep): picture sheet of the 34 checked SFK-BP ball moments. Free runner (needs the clip from R2).
Green ring = true ball, red cross = our pick, yellow dots = finder guesses, blue = player feet.
Output: results/qa/ballmisses/sheet_*.jpg + index.json. Claude sorts the misses by eye into causes."""
import os, json, cv2, numpy as np
M = {int(k): v for k, v in json.load(open("results/picker/moments34.json")).items()}
src = os.environ.get("LOCAL_CLIP") or os.environ["R2_PUBLIC_URL"].rstrip("/") + "/SFKBP1109_s1200/video.mp4"
cap = cv2.VideoCapture(src); print("frames", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), flush=True)
OUT = "results/qa/ballmisses"; os.makedirs(OUT, exist_ok=True); tiles = []; index = {}
for k in sorted(M):
    m = M[k]; cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
    if not ok: print("no frame", k); continue
    gx, gy = m["truth"]; hit = m["pick"] is not None and np.hypot(m["pick"][0] - gx, m["pick"][1] - gy) <= 30
    has = any(np.hypot(x - gx, y - gy) <= 30 for x, y, _ in m["guesses"])
    status = "right" if hit else ("picked other" if has else "no guess on ball")
    for px, py in m["players"]: cv2.circle(f, (int(px), int(py)), 5, (255, 120, 0), -1)
    for x, y, c in m["guesses"]: cv2.circle(f, (int(x), int(y)), 4, (0, 255, 255), -1)
    if m["pick"]: p = tuple(int(v) for v in m["pick"]); cv2.drawMarker(f, p, (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    cv2.circle(f, (int(gx), int(gy)), 16, (0, 255, 0), 2)
    x0 = int(min(max(gx - 200, 0), f.shape[1] - 400)); y0 = int(min(max(gy - 150, 0), f.shape[0] - 300))
    crop = f[y0:y0 + 300, x0:x0 + 400].copy()
    small = cv2.resize(f, (400, 225)); tile = np.vstack([crop, small])
    cv2.putText(tile, f"{k} {status}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    tiles.append((status, tile)); index[k] = status
tiles.sort(key=lambda t: t[0] == "right")
for s in range(0, len(tiles), 8):
    grp = [t for _, t in tiles[s:s + 8]]
    while len(grp) % 4: grp.append(np.zeros_like(grp[0]))
    rows = [np.hstack(grp[r:r + 4]) for r in range(0, len(grp), 4)]
    cv2.imwrite(f"{OUT}/sheet_{s // 8}.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
json.dump(index, open(f"{OUT}/index.json", "w"), indent=1); print(index)
