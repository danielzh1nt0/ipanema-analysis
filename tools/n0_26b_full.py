"""N0-26b (26 Oct, free runner): bigger frames (1280x720, one every 2 s) around the first two Veo goals of the full AIK match
and their kick-offs, to see where the ball goes in, who celebrates and who kicks off (the 640x360 sheets of
tools/n0_26b_frames.py are too small for that). Blind: nothing from the pipeline drawn.
    R2_PUBLIC_URL=... python tools/n0_26b_full.py        (LOCAL_VIDEO=video.mp4 for a dry run)
-> results/free/n0-26b/full/<window>_t<sec>.jpg"""
import os, cv2

MATCH = "p15u-vs-aik-2026-09-21-bd09"; OUT = "results/free/n0-26b/full"
WINDOWS = [("g2762", 2754, 2812, 2), ("g3128", 3122, 3172, 2)]


def main(windows=WINDOWS):
    os.makedirs(OUT, exist_ok=True)
    src = os.environ.get("LOCAL_VIDEO") or f"{os.environ['R2_PUBLIC_URL'].rstrip('/')}/{MATCH}/video.mp4"
    cap = cv2.VideoCapture(src)
    if not cap.isOpened(): raise SystemExit("video not reachable")
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97; made = 0
    for name, a, b, s in windows:
        for t in range(a, b + 1, s):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, fr = cap.read()
            if not ok: continue
            im = cv2.resize(fr, (1280, 720), interpolation=cv2.INTER_AREA)
            cv2.putText(im, f"{name} t {t} s", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
            cv2.putText(im, f"{name} t {t} s", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imwrite(f"{OUT}/{name}_t{t}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 78]); made += 1
    print(made, "frames ->", OUT, flush=True)
    return made


if __name__ == "__main__":
    main()
