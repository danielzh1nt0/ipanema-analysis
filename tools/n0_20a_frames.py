"""N0-20a (20 Oct, free runner): picture sheets of the app's AIK clip at its Veo shots/goal (results/qa/n0-20a), so the
shifted Veo times and the teams can be checked by eye: per event 12 frames from t to t+27.5 s (every 2.5 s), plus the
stretch after the goal (kick-off). Frames are 640x360, time printed on each.
    R2_PUBLIC_URL=... python tools/n0_20a_frames.py      (or LOCAL_CLIP=clip.mp4 for a dry run)"""
import os, json, urllib.request, numpy as np, cv2
CLIP = "p15u-vs-aik-2026-09-21-bd09_s2520"; OUT = os.environ.get("N0_20A_OUT", "results/free/n0-20a")
EVENTS = {"shot_012": 12.0, "shot_095": 95.0, "goal_242": 236.0, "after_goal_265": 265.0}   # goal sheet starts 6 s early
src = os.environ.get("LOCAL_CLIP")
if not src:
    src = "/tmp/n0_20a_clip.mp4"; url = os.environ["R2_PUBLIC_URL"].rstrip("/") + f"/{CLIP}/video.mp4"
    print("downloading", CLIP, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(src, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)
cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"clip {n} frames at {fps:.2f} fps ({n / fps:.1f} s)", flush=True)
os.makedirs(OUT, exist_ok=True); made = {}
for name, t0 in EVENTS.items():
    tiles = []
    for j in range(12):
        t = t0 + 2.5 * j; k = int(round(t * fps))
        if k >= n: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, fr = cap.read()
        if not ok: break
        im = cv2.resize(fr, (640, 360), interpolation=cv2.INTER_AREA)
        txt = f"clip {t:.1f} s  (match {2520 + t:.0f} s)"
        cv2.putText(im, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4); cv2.putText(im, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(im)
    while len(tiles) % 3: tiles.append(np.zeros((360, 640, 3), np.uint8))
    if not tiles: continue
    sheet = np.vstack([np.hstack(tiles[r:r + 3]) for r in range(0, len(tiles), 3)])
    cv2.imwrite(f"{OUT}/{name}.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]); made[name] = len(tiles)
json.dump({"clip": CLIP, "fps": fps, "frames": n, "sheets": made}, open(f"{OUT}/sheets.json", "w"), indent=1)
print("sheets:", made, flush=True)
