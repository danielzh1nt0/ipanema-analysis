"""N0-26b (26 Oct, free runner): which end does each team attack in the full AIK match? Picture sheets of the kick-offs and
the Veo goals, nothing from the pipeline drawn (blind): whole frames scaled to 640x360, one every STEP s, 12 per sheet,
labelled with the video time. Used to settle a conflict: periods/<match>.json says the dark team (SFK) attacks right in the
first half (by eye, 4 Oct), while the free-pipeline export's kick-off formations put the dark team in the RIGHT half at all 9
kick-offs (results/qa/n0-26b/README.md).
    R2_PUBLIC_URL=... python tools/n0_26b_frames.py        (LOCAL_VIDEO=video.mp4 for a dry run)
-> results/free/n0-26b/<window>_NN.jpg + index.json"""
import os, json, cv2, numpy as np

MATCH = "p15u-vs-aik-2026-09-21-bd09"; OUT = "results/free/n0-26b"; TW, TH, COLS, ROWS = 640, 360, 3, 4
# (name, start s, end s, step s): first-half kick-off, the Veo goals (reference/veo_highlights_<match>.txt) up to their
# kick-off, second-half kick-off
WINDOWS = [("ko1", 356, 446, 3), ("g2762", 2740, 2836, 3), ("g3128", 3106, 3202, 3), ("ko2", 3936, 4020, 3),
           ("g4257", 4235, 4331, 3), ("g4962", 4940, 5036, 3), ("g5625", 5603, 5699, 3), ("g5916", 5894, 5990, 3),
           ("g6268", 6246, 6342, 3)]


def times(w):
    _, a, b, s = w
    return [round(a + i * s, 1) for i in range(int((b - a) // s) + 1)]


def label(im, txt):
    cv2.putText(im, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
    cv2.putText(im, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


def sheets(tiles, name):
    out = []
    per = COLS * ROWS
    for s in range(0, len(tiles), per):
        chunk = tiles[s:s + per]
        chunk = chunk + [np.zeros((TH, TW, 3), np.uint8)] * (per - len(chunk))
        rows = [np.hstack(chunk[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]
        p = f"{OUT}/{name}_{s // per:02d}.jpg"; cv2.imwrite(p, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82]); out.append(p)
    return out


def main(windows=WINDOWS):
    os.makedirs(OUT, exist_ok=True)
    src = os.environ.get("LOCAL_VIDEO") or f"{os.environ['R2_PUBLIC_URL'].rstrip('/')}/{MATCH}/video.mp4"
    cap = cv2.VideoCapture(src)
    if not cap.isOpened(): raise SystemExit("video not reachable")
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97; n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print("fps", fps, "frames", n, flush=True)
    index = {}
    for w in windows:
        tiles = []
        for t in times(w):
            k = int(round(t * fps))
            if n and k >= n: break
            cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, fr = cap.read()
            if not ok: continue
            im = cv2.resize(fr, (TW, TH), interpolation=cv2.INTER_AREA); label(im, f"{w[0]}  t {t:.0f} s"); tiles.append(im)
        index[w[0]] = {"times": times(w)[:len(tiles)], "sheets": sheets(tiles, w[0]) if tiles else []}
        print(w[0], len(tiles), "frames", flush=True)
    cap.release()
    json.dump({"match": MATCH, "windows": index}, open(f"{OUT}/index.json", "w"), indent=1)
    return index


if __name__ == "__main__":
    main()
