"""N0-28a (28 Oct, free runner): blind picture sheets of the app clips' reactions after a loss (results/qa/n0-28a/moments.json
from tools/n0_28a_clips.py). Each sheet: 6 tiles at 0, 0.4, ... 2.0 s after the ball was won, each the same full-resolution
640x360 crop around the marked player's path (the camera itself pans) (yellow ring = the player of the team that lost it, red ring = the app's ball pick;
when the ball is outside the crop a red arrow on the edge points to it). Question when grading: over these 2 s, does the
yellow player run at the ball (pressed), move toward it slowly (jogged) or stand / walk / go the other way (stood)?
The app's labels (old or new) are NOT on the sheet; they are in results/qa/n0-28a/key_SEALED.json.
    R2_PUBLIC_URL=... python tools/n0_28a_frames.py      (LOCAL_CLIP_SFK / LOCAL_CLIP_AIK = local videos for a dry run)"""
import os, sys, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from n0_27b_frames import fetch, put, TW, TH, COLS
M = json.load(open("results/qa/n0-28a/moments.json")); OUT = "results/free/n0-28a"


def tile(fr, tl, centre):
    H0, W0 = fr.shape[:2]
    if (W0, H0) != (1920, 1080): fr = cv2.resize(fr, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    cx = int(min(max(centre[0], TW / 2), 1920 - TW / 2)); cy = int(min(max(centre[1], TH / 2), 1080 - TH / 2))
    x0, y0 = cx - TW // 2, cy - TH // 2; im = fr[y0:y0 + TH, x0:x0 + TW].copy()
    if tl.get("player_px"): cv2.circle(im, (int(tl["player_px"][0] - x0), int(tl["player_px"][1] - y0)), 20, (0, 255, 255), 2, cv2.LINE_AA)
    b = tl.get("ball_px")
    if b:
        bx, by = b[0] - x0, b[1] - y0
        if 0 <= bx < TW and 0 <= by < TH: cv2.circle(im, (int(bx), int(by)), 12, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            c = np.array([TW / 2, TH / 2]); v = np.array([bx, by]) - c; s = min((TW / 2 - 20) / max(abs(v[0]), 1e-6), (TH / 2 - 20) / max(abs(v[1]), 1e-6))
            tip = c + v * s; tail = c + v * s * 0.8
            cv2.arrowedLine(im, (int(tail[0]), int(tail[1])), (int(tip[0]), int(tip[1])), (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.4)
    return im


def main():
    os.makedirs(OUT, exist_ok=True); made = {}; rc = 0
    for short, clip in (("sfk", "SFKBP1109_s1200"), ("aik", "p15u-vs-aik-2026-09-21-bd09_s2520")):
        ms = [m for m in M if m["clip"] == clip]
        if not ms: continue
        cap = cv2.VideoCapture(fetch(clip, short)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); print("==", short, len(ms), "moments,", n, "frames", flush=True)
        os.makedirs(f"{OUT}/{short}", exist_ok=True)
        for m in ms:
            tiles = []
            pts = [x["player_px"] for x in m["tiles"] if x.get("player_px")]   # ONE crop for the sheet, so the run shows
            fixed = list(np.mean(pts, axis=0)) if pts else (m["tiles"][0].get("ball_px") or (960, 540))
            for tl in m["tiles"]:
                k = tl["frame"]
                if k >= n: break
                cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, fr = cap.read()
                if not ok: break
                im = tile(fr, tl, fixed)
                put(im, f"{m['id']}  +{tl['t'] - m['tiles'][0]['t']:.1f} s after the win", 22); put(im, f"yellow = player of kit {m['team']} (lost it), red = ball", 44)
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
