"""V1 (5 Oct, worker), free GitHub runner, no Modal: picture sheets for finding where each Veo shot was struck.

Reads a spec (default results/review/v1/spec.json):
  {"jobs": [{"name": "bp_610_p1", "src_key": "SFKBP1109/video.mp4", "times": [612, 613, ...],
             "crop": null | [x0, y0, w, h] (1920x1080 pixels), "tile_w": 640, "per_sheet": 12, "cols": 3}]}
For every job: one tile per time (full frame or a full-resolution crop), a 100-px grid in 1920x1080 frame pixels (red
every 500 px, small labels on the crop edges so a mark can be read back in frame pixels), time label, sheets of
`per_sheet` tiles -> results/free/v1/<name>_<k>.jpg, plus results/free/v1/index.json.
Video from R2 (R2_PUBLIC_URL) or a local path when src_key is a file (dry run / tests).
    python tools/v1_frames.py [spec.json]"""
import os, sys, json, cv2, numpy as np

OUT = "results/free/v1"

def grid(f, x0=0, y0=0):
    """draw the frame-pixel grid on a 1920x1080 frame (or a crop starting at x0, y0)"""
    h, w = f.shape[:2]
    for X in range((x0 // 100) * 100, x0 + w + 1, 100):
        if X < x0: continue
        cv2.line(f, (X - x0, 0), (X - x0, h), (0, 0, 255) if X % 500 == 0 else (0, 255, 255), 1)
    for Y in range((y0 // 100) * 100, y0 + h + 1, 100):
        if Y < y0: continue
        cv2.line(f, (0, Y - y0), (w, Y - y0), (0, 0, 255) if Y % 500 == 0 else (0, 255, 255), 1)
    return f

def labels(t, x0, y0, w, h, scale, txt):
    """edge labels in frame pixels drawn on the (scaled) tile"""
    out = []
    for X in range(((x0 + 99) // 100) * 100, x0 + w, 100): out.append((str(X), (int((X - x0) * scale) + 2, 12)))
    for Y in range(((y0 + 99) // 100) * 100, y0 + h, 100): out.append((str(Y), (2, int((Y - y0) * scale) - 2)))
    return out

def tile(frame, t, crop, tile_w, title):
    if frame is None: frame = np.zeros((1080, 1920, 3), np.uint8)
    if frame.shape[:2] != (1080, 1920): frame = cv2.resize(frame, (1920, 1080))
    x0, y0, w, h = crop or (0, 0, 1920, 1080)
    x0 = max(0, min(1920 - w, int(x0))); y0 = max(0, min(1080 - h, int(y0)))
    c = grid(frame[y0:y0 + h, x0:x0 + w].copy(), x0, y0)
    s = tile_w / w; g = cv2.resize(c, (tile_w, int(round(h * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
    for txt, p in labels(t, x0, y0, w, h, s, None): cv2.putText(g, txt, p, 0, 0.35, (255, 255, 255), 1)
    cv2.rectangle(g, (0, g.shape[0] - 20), (min(tile_w, 330), g.shape[0]), (0, 0, 0), -1)
    cv2.putText(g, title, (4, g.shape[0] - 6), 0, 0.5, (255, 255, 255), 1)
    return g

def run(spec, base=None):
    base = (base if base is not None else os.environ.get("R2_PUBLIC_URL", "")).rstrip("/")
    os.makedirs(OUT, exist_ok=True); index = {}
    for job in spec["jobs"]:
        src = job["src_key"] if os.path.exists(job["src_key"]) else f"{base}/{job['src_key']}"
        cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
        tiles = []
        for t in job["times"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read()
            tiles.append(tile(f if ok else None, t, job.get("crop"), job.get("tile_w", 640), f"{job['name']} t={t:.1f}s"))
        per, cols = job.get("per_sheet", 12), job.get("cols", 3); sheets = []
        for i in range(0, len(tiles), per):
            ts = tiles[i:i + per]
            while len(ts) % cols: ts.append(np.zeros_like(tiles[0]))
            rows = [np.hstack(ts[r:r + cols]) for r in range(0, len(ts), cols)]
            p = f"{OUT}/{job['name']}_{i // per}.jpg"; cv2.imwrite(p, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88]); sheets.append(p)
        index[job["name"]] = {"src_key": job["src_key"], "times": job["times"], "crop": job.get("crop"), "fps": fps, "sheets": sheets}
        print(job["name"], len(tiles), "tiles", flush=True)
    json.dump(index, open(f"{OUT}/index.json", "w"), indent=1)
    return index

if __name__ == "__main__":
    run(json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/review/v1/spec.json")))
