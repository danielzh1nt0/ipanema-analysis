"""N0-01e (1 Nov, worker, free runner, $0, nothing on Modal): AIK's kick-offs after every goal, by eye.

Open question (Waiting for Daniel, N0-26b): which way does each team attack in the AIK match?
periods/p15u-vs-aik-2026-09-21-bd09.json says dark SFK attacks RIGHT in the first half (4 Oct, judged from where the
goals were scored); the two half kick-offs (law-bound: each team stands in the half it defends) say LEFT.
After every goal there is a kick-off too, so the 7 Veo goals give 7 more law-bound line-ups, and the team at the ball
is the team that conceded (-> who scored, checkable against Veo's final score 1-6).
Also the 84' penalty (Veo: AIK penalty, goal 1-6): which end, and who takes it.

    python tools/n0_01e_aik_kickoffs.py              # on the free runner (R2_PUBLIC_URL set)
    SRC=some.mp4 python tools/n0_01e_aik_kickoffs.py # dry run on a local video

-> results/free/n0_01e/<window>_<t>.jpg (1280 px wide, time stamped) + sheet_<window>.jpg + times.json
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n0_31c_sfk_periods as C

M = "p15u-vs-aik-2026-09-21-bd09"
GOALS = [2762, 3128, 4257, 4962, 5625, 5916, 6268]       # reference/veo_highlights_<M>.txt (clip start times)
WINDOWS = [(f"ko{g}", g + 25.0, g + 90.0, 5.0) for g in GOALS] + [("pen6268", 6250.0, 6300.0, 5.0)]
OUT = os.environ.get("OUT", "results/free/n0_01e")
WIDTH = 1280                                              # kick-off line-ups need more than the 960 px of N0-31d


def main():
    import cv2, numpy as np
    C.M = M
    C.W = WIDTH
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
