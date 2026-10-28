"""Open a match video over the network (R2) with retries.

N0-28c (28 Oct): the 28 Oct nightly lost the Djursholm player check because R2 stalled for 30 s while OpenCV read the
mp4 header ('Stream timeout ... Stream ends prematurely', 0 frames); the same file opened fine 25 min earlier. One stall
should not cost a night's check: re-open up to `tries` times, waiting longer each time, and also re-read the piece if
the stream dies part-way (fewer than `min_share` of the wanted frames)."""
import os, time

# N0-28c: R2 sometimes stops sending for 30 s+ (no error), and OpenCV gives up after 30 s. ffmpeg's own HTTP options
# time a silent read out after 10 s and reconnect at the same byte; with a 2-min open/read budget a test server that
# stalls 3 times now gives 100/100 frames (plain OpenCV: 0, the runner's 'Stream ends prematurely' failure).
HTTP_OPTIONS = "rw_timeout;10000000|reconnect;1|reconnect_on_network_error;1|reconnect_streamed;1|reconnect_delay_max;30"
TIMEOUT_MS = 120000


def honours_ffmpeg_options(cv):
    """N0-28d: OpenCV 4.11.0 has the OPENCV_FFMPEG_CAPTURE_OPTIONS check upside down (cap_ffmpeg_impl.hpp line 1140:
    `if(!options.empty())` skips the options), so the reconnect settings above are silently dropped. 4.10 and 4.12+
    are fine. With numpy<2 pip picks 4.11.0.86, which is what the runner had: a stall sat silent for the whole 2 min
    (reproduced locally: 4.11 -> 0 frames after 120 s, 4.10 -> reconnect after 10 s, 300/300 frames).
    The workflows now pin opencv-python-headless==4.10.0.84."""
    return not str(getattr(cv, "__version__", "")).startswith("4.11.")


_warned = [False]


def capture(cv, src, log=print):
    """cv.VideoCapture with stall-proof HTTP settings for a URL (plain open for a local file or a stand-in cv)."""
    if str(src).startswith(("http://", "https://")) and hasattr(cv, "CAP_FFMPEG") and hasattr(cv, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        if not honours_ffmpeg_options(cv) and not _warned[0]:
            _warned[0] = True
            log(f"WARNING: OpenCV {cv.__version__} ignores OPENCV_FFMPEG_CAPTURE_OPTIONS - an R2 stall will not reconnect (pin opencv-python-headless==4.10.0.84)")
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", HTTP_OPTIONS)
        return cv.VideoCapture(src, cv.CAP_FFMPEG, [cv.CAP_PROP_OPEN_TIMEOUT_MSEC, TIMEOUT_MS, cv.CAP_PROP_READ_TIMEOUT_MSEC, TIMEOUT_MS])
    return cv.VideoCapture(src)


def open_clip(src, cv, tries=3, wait_s=20.0, min_frames=100, sleep=time.sleep, log=print):
    """-> (cap, fps, n_all, attempts). cap is None when every try gave fewer than min_frames frames."""
    for a in range(1, tries + 1):
        cap = capture(cv, src, log=log)
        fps = cap.get(cv.CAP_PROP_FPS) or 29.97
        n_all = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
        if n_all >= min_frames:
            return cap, fps, n_all, a
        cap.release()
        if a < tries:
            log(f"clip open try {a}/{tries}: {n_all} frames - waiting {wait_s * a:.0f} s and trying again")
            sleep(wait_s * a)
    return None, 29.97, 0, tries


def read_piece(src, cv, k0, n, out_path, tries=3, wait_s=20.0, min_share=0.95, sleep=time.sleep, log=print, cap=None):
    """Write frames k0 .. k0+n-1 to out_path (mp4v) -> (frames written, fps, attempts). Frames go straight to disk (20 s of
    1080p is ~3.6 GB in memory). Re-opens the clip when the stream dies part-way; keeps the longest read if every try
    comes up short. `cap` (an already open capture) is used for the first try."""
    import os
    best, fps_best = 0, 29.97
    for a in range(1, tries + 1):
        fps = 29.97
        if cap is None:
            cap, fps, _, _ = open_clip(src, cv, tries=1, log=log)
        else:
            fps = cap.get(cv.CAP_PROP_FPS) or 29.97
        got, w, tmp = 0, None, f"{out_path}.try{a}.mp4"
        if cap is not None:
            cap.set(cv.CAP_PROP_POS_FRAMES, k0)
            for _ in range(n):
                ok, f = cap.read()
                if not ok:
                    break
                if w is None:
                    w = cv.VideoWriter(tmp, cv.VideoWriter_fourcc(*"mp4v"), fps, (f.shape[1], f.shape[0]))
                w.write(f); got += 1
            cap.release()
        if w is not None:
            w.release()
        cap = None
        if got and not os.path.exists(tmp):          # N0-28d dry run: a writer that never opened left no file -> crash; count the try as 0
            log(f"piece read try {a}/{tries}: {got} frames read but the writer made no file - counting 0")
            got = 0
        if got > best:
            best, fps_best = got, fps
            os.replace(tmp, out_path)
        elif os.path.exists(tmp):
            os.remove(tmp)
        if got >= min_share * n:
            return got, fps, a
        if a < tries:
            log(f"piece read try {a}/{tries}: {got}/{n} frames - waiting {wait_s * a:.0f} s and trying again")
            sleep(wait_s * a)
    return best, fps_best, tries


def local_copy(src, dest="/tmp/clip.mp4", max_bytes=1.5e9, run=None, log=print):
    """N0-28c: the SFK-BP clip (300 MB) never read over the stream on the runner - 4 of 4 nightly runs on 28 Oct got 2-23
    frames of the 599 (OpenCV reads from the start up to the 60-s mark and R2 cuts the stream after ~70 MB), so its
    player row was 0 players / frame. A clip under max_bytes is downloaded whole first (curl with retries, seconds on
    the runner). -> the path to read (dest, or src when it is local / too big / the download failed)."""
    import subprocess
    if not str(src).startswith(("http://", "https://")): return src
    run = run or subprocess.run
    h = run(["curl", "-sSfLI", "--retry", "3", src], capture_output=True, text=True)
    size = None
    for ln in (h.stdout or "").splitlines():
        if ln.lower().startswith("content-length:"):
            try: size = int(ln.split(":", 1)[1].strip())
            except ValueError: pass
    if h.returncode != 0 or size is None or size > max_bytes:
        log(f"clip streamed ({'size unknown' if size is None else f'{size / 1e6:.0f} MB'})"); return src
    r = run(["curl", "-sSfL", "--retry", "5", "--retry-all-errors", "--speed-limit", "10000", "--speed-time", "20", "-o", dest, src], capture_output=True, text=True)
    import os
    if r.returncode != 0 or not os.path.exists(dest) or os.path.getsize(dest) != size:
        log(f"clip download failed ({(r.stderr or '').strip()[:120]}) - streaming instead"); return src
    log(f"clip downloaded whole ({size / 1e6:.0f} MB)"); return dest
