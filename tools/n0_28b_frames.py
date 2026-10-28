"""N0-28b (28 Oct, free runner): blind picture sheets for the stoppage spans the new sequence split cuts at, plus
control windows (results/qa/n0-28b/moments.json from tools/n0_28b_clips.py --write-moments). Each sheet: 8 whole-frame
tiles (640x360): 1 s before the window, 6 across it, 1 s after. Question when grading: in each tile, is play on or
stopped (ball out / players waiting for a restart / walking)? Which windows are the app's stoppages is NOT on the sheet
(results/qa/n0-28b/key_SEALED.json).
    R2_PUBLIC_URL=... python tools/n0_28b_frames.py      (LOCAL_CLIP_SFK / LOCAL_CLIP_AIK = local videos for a dry run)"""
import os, sys, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from n0_27b_frames import fetch, put
M = json.load(open("results/qa/n0-28b/moments.json")); OUT = "results/free/n0-28b"; TW, TH, COLS = 640, 360, 4


def main():
    os.makedirs(OUT, exist_ok=True); made = {}; rc = 0
    for short, clip in (("sfk", "SFKBP1109_s1200"), ("aik", "p15u-vs-aik-2026-09-21-bd09_s2520")):
        ms = [m for m in M if m["clip"] == clip]
        if not ms: continue
        cap = cv2.VideoCapture(fetch(clip, short)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        print("==", short, len(ms), "windows,", n, "frames", fps, "fps", flush=True)
        for m in ms:
            tiles = []
            for t in m["times"]:
                k = int(round(t * fps))
                if k >= n: break
                cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, fr = cap.read()
                if not ok: break
                im = cv2.resize(fr, (TW, TH), interpolation=cv2.INTER_AREA); put(im, f"{m['id']}  t={t:.1f} s", 22); tiles.append(im)
            if not tiles: rc = 1; continue
            real = len(tiles)
            while len(tiles) % COLS: tiles.append(np.zeros((TH, TW, 3), np.uint8))
            sheet = np.vstack([np.hstack(tiles[r:r + COLS]) for r in range(0, len(tiles), COLS)])
            cv2.imwrite(f"{OUT}/{m['id']}.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]); made[m["id"]] = real
    json.dump({"sheets": made}, open(f"{OUT}/sheets.json", "w"), indent=1)
    print("sheets:", made, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
