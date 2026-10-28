"""Open a match video over the network (R2) with retries.

N0-28c (28 Oct): the 28 Oct nightly lost the Djursholm player check because R2 stalled for 30 s while OpenCV read the
mp4 header ('Stream timeout ... Stream ends prematurely', 0 frames); the same file opened fine 25 min earlier. One stall
should not cost a night's check: re-open up to `tries` times, waiting longer each time, and also re-read the piece if
the stream dies part-way (fewer than `min_share` of the wanted frames)."""
import time


def open_clip(src, cv, tries=3, wait_s=20.0, min_frames=100, sleep=time.sleep, log=print):
    """-> (cap, fps, n_all, attempts). cap is None when every try gave fewer than min_frames frames."""
    for a in range(1, tries + 1):
        cap = cv.VideoCapture(src)
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
