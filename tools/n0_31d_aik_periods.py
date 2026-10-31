"""N0-31d (31 Oct, worker, free runner, $0, nothing on Modal): check AIK's kick-offs and full time by eye.

periods/p15u-vs-aik-2026-09-21-bd09.json has only the first-half END checked by eye (3225 s, D6). The first-half
kick-off (366 s) and second-half kick-off (3950 s) came from a detector, full time (6899 s) from the video. Side
findings say all three are off (N0-30a / N0-26b: kick-offs ~428-446 and ~3984-4005 s; N0-30b: the pitch is empty
after the whistle from ~6777 s, play at 6679 s), but nobody looked at frames close together. Same method as
N0-31c (SFK-BP): frames every 5-10 s around each boundary, graded by eye.

    python tools/n0_31d_aik_periods.py              # on the free runner (R2_PUBLIC_URL set)
    SRC=some.mp4 python tools/n0_31d_aik_periods.py # dry run on a local video

-> results/free/n0_31d/<window>_<t>.jpg (960 px wide, time stamped) + sheet_<window>.jpg + times.json
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n0_31c_sfk_periods as C

M = "p15u-vs-aik-2026-09-21-bd09"
WINDOWS = [("ko1", 360.0, 480.0, 5.0),      # first-half kick-off: file 366, side findings 428-446
           ("ht", 3200.0, 3250.0, 10.0),    # first-half end 3225 (D6 by eye) - control
           ("ko2", 3940.0, 4040.0, 5.0),    # second-half kick-off: file 3950, side findings 3984-4005
           ("ft", 6640.0, 6800.0, 10.0)]    # full time: file 6899, N0-30b play at 6679, empty pitch from 6777
OUT = os.environ.get("OUT", "results/free/n0_31d")


def main():
    import cv2, numpy as np
    C.M = M                                   # times.json names the right match
    src = os.environ.get("SRC")
    if not src:
        base = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
        if not base:
            raise SystemExit("set R2_PUBLIC_URL or SRC")
        from clipopen import local_copy        # whole video (R2 cuts long streams, N0-28c)
        src = local_copy(f"{base}/{M}/video.mp4", dest="/tmp/aik_video.mp4", max_bytes=12e9)
    wins = C.parse_windows(os.environ.get("WIN")) if os.environ.get("WIN") else WINDOWS
    C.cut(src, C.window_times(wins), OUT, cv2, np)


if __name__ == "__main__":
    main()
