"""N0-20a (20 Oct, free runner): picture sheets of the app's AIK clip at its Veo shots/goal (results/qa/n0-20a), so the
shifted Veo times and the teams can be checked by eye: per event 12 frames from t to t+27.5 s (every 2.5 s), plus the
stretch after the goal (kick-off). Frames are 640x360, time printed on each.
    R2_PUBLIC_URL=... python tools/n0_20a_frames.py      (or LOCAL_CLIP=clip.mp4 for a dry run)"""
import os, sys, json, urllib.request, numpy as np, cv2
CLIP = "p15u-vs-aik-2026-09-21-bd09_s2520"; OUT = os.environ.get("N0_20A_OUT", "results/free/n0-20a")
EVENTS = {"shot_012": 12.0, "shot_095": 95.0, "goal_242": 236.0, "after_goal_265": 265.0}   # goal sheet starts 6 s early
# 20 Oct 04:20 (part 2): denser, bigger frames where the goal should be: N0_20A_EVENTS='{"g1": 254}' N0_20A_STEP=1
# N0_20A_N=12 N0_20A_W=960 N0_20A_COLS=2 (defaults = the part-1 sheets)
# "python tools/n0_20a_frames.py dense" = the part-2 preset (the free runner can't pass env vars): 1 s apart, 960 px,
# 254-301 s in 4 sheets of 12, into results/free/n0-20a/dense
if sys.argv[1:2] == ["dense"]:
    os.environ.setdefault("N0_20A_EVENTS", json.dumps({"dense_254": 254, "dense_266": 266, "dense_278": 278, "dense_290": 290}))
    os.environ.setdefault("N0_20A_STEP", "1"); os.environ.setdefault("N0_20A_W", "960"); os.environ.setdefault("N0_20A_COLS", "2")
    OUT = os.environ.get("N0_20A_OUT", "results/free/n0-20a/dense")
if os.environ.get("N0_20A_EVENTS"): EVENTS = {k: float(v) for k, v in json.loads(os.environ["N0_20A_EVENTS"]).items()}
STEP = float(os.environ.get("N0_20A_STEP", 2.5)); NPER = int(os.environ.get("N0_20A_N", 12))
TW = int(os.environ.get("N0_20A_W", 640)); TH = TW * 9 // 16; COLS = int(os.environ.get("N0_20A_COLS", 3))
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
    for j in range(NPER):
        t = t0 + STEP * j; k = int(round(t * fps))
        if k >= n: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, fr = cap.read()
        if not ok: break
        im = cv2.resize(fr, (TW, TH), interpolation=cv2.INTER_AREA)
        txt = f"clip {t:.1f} s  (match {2520 + t:.0f} s)"
        cv2.putText(im, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4); cv2.putText(im, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(im)
    real = len(tiles)   # frames shown (the padding tiles are black)
    while len(tiles) % COLS: tiles.append(np.zeros((TH, TW, 3), np.uint8))
    if not tiles: continue
    sheet = np.vstack([np.hstack(tiles[r:r + COLS]) for r in range(0, len(tiles), COLS)])
    cv2.imwrite(f"{OUT}/{name}.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]); made[name] = real
json.dump({"clip": CLIP, "step": STEP, "tile_w": TW, "fps": fps, "frames": n, "sheets": made}, open(f"{OUT}/sheets.json", "w"), indent=1)
print("sheets:", made, flush=True)
