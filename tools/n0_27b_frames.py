"""N0-27b (27 Oct, free runner): picture sheets of a seeded sample of the app clips' LOST passes that now get a pass
quality (results/qa/n0-27b/moments.json from tools/n0_27b_clips.py). Each sheet: 6 tiles at -0.5, 0, +0.4, +0.8, +1.2,
+1.6 s (2nd tile = the passer's last frame on the ball), each a full-resolution 640x360 crop around the passer and the
intended mate (no scaling), app ball pick = red ring, yellow = passer's feet, green = the intended mate (the passer's
team-mate nearest where the ball was won), magenta = the opponent who won it. Questions when grading: is it a pass that
the other team won? is green a plausible target? was an opponent in the straight lane yellow -> green at the kick?
Whether the app calls the lane blocked or open is NOT on the sheet.
    R2_PUBLIC_URL=... python tools/n0_27b_frames.py      (LOCAL_CLIP_SFK / LOCAL_CLIP_AIK = local videos for a dry run)"""
import os, sys, json, urllib.request, numpy as np, cv2
M = json.load(open("results/qa/n0-27b/moments.json")); OUT = "results/free/n0-27b"; TW, TH, COLS = 640, 360, 3


def fetch(clip, short):
    src = os.environ.get(f"LOCAL_CLIP_{short.upper()}")
    if src: return src
    src = f"/tmp/n0_27b_{short}.mp4"; url = os.environ["R2_PUBLIC_URL"].rstrip("/") + f"/{clip}/video.mp4"
    print("downloading", clip, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(src, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)
    return src


def put(im, txt, y):
    cv2.putText(im, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4); cv2.putText(im, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def tile(fr, tl, sx, sy, centre, mid):
    """640x360 crop of the frame (scaled to 1920x1080 coordinates) around centre, marks drawn"""
    H0, W0 = fr.shape[:2]
    if (W0, H0) != (1920, 1080): fr = cv2.resize(fr, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    cx = int(min(max(centre[0], TW / 2), 1920 - TW / 2)); cy = int(min(max(centre[1], TH / 2), 1080 - TH / 2))
    x0, y0 = cx - TW // 2, cy - TH // 2; im = fr[y0:y0 + TH, x0:x0 + TW].copy()
    for key, col in (("passer_px", (0, 255, 255)), ("mate_px", (0, 255, 0)), ("winner_px", (255, 0, 255))):
        if tl.get(key): cv2.circle(im, (int(tl[key][0] - x0), int(tl[key][1] - y0)), 18, col, 2, cv2.LINE_AA)
    if tl.get("ball_px"): cv2.circle(im, (int(tl["ball_px"][0] - x0), int(tl["ball_px"][1] - y0)), 12, (0, 0, 255), 1, cv2.LINE_AA)
    return im


def main():
    os.makedirs(OUT, exist_ok=True); made = {}; rc = 0
    for short, clip in (("sfk", "SFKBP1109_s1200"), ("aik", "p15u-vs-aik-2026-09-21-bd09_s2520")):
        ms = [m for m in M if m["clip"] == clip]
        if not ms: continue
        cap = cv2.VideoCapture(fetch(clip, short)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); print("==", short, len(ms), "moments,", n, "frames", flush=True)
        os.makedirs(f"{OUT}/{short}", exist_ok=True)
        for m in ms:
            tiles = []; last = None
            for tl in m["tiles"]:
                k = tl["frame"]
                if k >= n: break
                cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, fr = cap.read()
                if not ok: break
                a, b = tl.get("passer_px"), tl.get("mate_px")
                centre = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] if a and b else (a or b or tl.get("ball_px") or last)
                if centre is None: centre = next((x.get("passer_px") or x.get("ball_px") for x in m["tiles"] if x.get("passer_px") or x.get("ball_px")), (960, 540))
                last = centre
                im = tile(fr, tl, 1, 1, centre, None)
                put(im, f"{m['id']}  t {tl['t']:.2f} s" + ("  <- kick" if tl is m["tiles"][1] else ""), 22); put(im, f"yellow passer (kit {m['team']}), green mate, magenta winner", 44)
                tiles.append(im)
            if not tiles: rc = 1; continue
            real = len(tiles)
            while len(tiles) % COLS: tiles.append(np.zeros((TH, TW, 3), np.uint8))
            sheet = np.vstack([np.hstack(tiles[r:r + COLS]) for r in range(0, len(tiles), COLS)])
            cv2.imwrite(f"{OUT}/{short}/{m['id']}.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]); made[m["id"]] = real
    json.dump({"sheets": made}, open(f"{OUT}/sheets.json", "w"), indent=1)
    print("sheets:", made, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
