"""N0-20b (20 Oct, free runner): picture sheets of the app's SFK-BP test clip at each candidate final-third entry
(results/qa/n0-20b/moments.json, built by tools/entrylab.py moments), to grade by eye which entries are real. Each sheet:
tiles every 0.5 s from 2 s before the first candidate to 1.5 s after the last, 640 px, time printed, the app's ball pick
as a red ring, the attacking team's 2/3 line (start of its final third) in yellow. Which rule counted which entry is NOT
on the sheet (moments_key.json, opened after grading).
    R2_PUBLIC_URL=... python tools/n0_20b_frames.py      (or LOCAL_CLIP=clip.mp4 for a dry run)"""
import os, sys, json, urllib.request, numpy as np, cv2
M = json.load(open(os.environ.get("N0_20B_MOMENTS", "results/qa/n0-20b/moments.json"))); CLIP = M["clip"]
OUT = os.environ.get("N0_20B_OUT", "results/free/n0-20b"); TW, COLS = 640, 3; TH = TW * 9 // 16
src = os.environ.get("LOCAL_CLIP")
if not src:
    src = "/tmp/n0_20b_clip.mp4"; url = os.environ["R2_PUBLIC_URL"].rstrip("/") + f"/{CLIP}/video.mp4"
    print("downloading", CLIP, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(src, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)
cap = cv2.VideoCapture(src); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W0 = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920
print(f"clip {n} frames, {W0:.0f} px wide", flush=True)
s = TW / 1920.0      # the moments file is in 1920x1080 pixels; tiles are TW wide whatever the clip's own size
os.makedirs(OUT, exist_ok=True); made = {}

def put(im, txt, y):
    cv2.putText(im, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4); cv2.putText(im, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

for m in M["moments"]:
    tiles = []
    for tl in m["tiles"]:
        k = tl["frame"]
        if k >= n: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, fr = cap.read()
        if not ok: break
        im = cv2.resize(fr, (TW, TH), interpolation=cv2.INTER_AREA)
        if tl.get("line_px"):
            (x1, y1), (x2, y2) = tl["line_px"]
            ok2, p1, p2 = cv2.clipLine((0, 0, TW, TH), (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)))
            if ok2: cv2.line(im, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
        if tl.get("ball_px"): cv2.circle(im, (int(tl["ball_px"][0] * s), int(tl["ball_px"][1] * s)), 9, (0, 0, 255), 1, cv2.LINE_AA)
        put(im, f"{m['id']}  t {tl['t']:.1f} s", 22); put(im, f"{m['team']} attacks {m['attacks']}", 44)
        tiles.append(im)
    real = len(tiles)
    if not tiles: continue
    while len(tiles) % COLS: tiles.append(np.zeros((TH, TW, 3), np.uint8))
    sheet = np.vstack([np.hstack(tiles[r:r + COLS]) for r in range(0, len(tiles), COLS)])
    cv2.imwrite(f"{OUT}/{m['id']}.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]); made[m["id"]] = real
json.dump({"clip": CLIP, "tile_w": TW, "frames": n, "sheets": made}, open(f"{OUT}/sheets.json", "w"), indent=1)
print("sheets:", made, flush=True)
