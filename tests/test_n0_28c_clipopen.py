"""N0-28c: the nightly player check re-tries an R2 clip that stalls (28 Oct Djursholm: 0 frames after a 30 s stream timeout)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tools.clipopen import open_clip, read_piece


class FakeCV:
    CAP_PROP_FPS, CAP_PROP_FRAME_COUNT, CAP_PROP_POS_FRAMES = 5, 7, 1

    def __init__(self, plan):            # plan: list of (frame_count, frames readable) per open
        self.plan, self.opens, self.written = list(plan), 0, {}

    def VideoCapture(self, src):
        n_all, readable = self.plan[min(self.opens, len(self.plan) - 1)]; self.opens += 1
        cv = self

        class Cap:
            pos, left = 0, readable
            def get(self, p): return 25.0 if p == cv.CAP_PROP_FPS else n_all
            def set(self, p, v): self.pos = v
            def read(self):
                if self.left <= 0: return False, None
                self.left -= 1; return True, np.zeros((4, 6, 3), np.uint8)
            def release(self): pass
        return Cap()

    def VideoWriter_fourcc(self, *a): return 0

    def VideoWriter(self, path, fourcc, fps, size):
        cv = self

        class W:
            k = 0
            def write(self, f): self.k += 1
            def release(self):
                open(path, "w").write("x" * self.k); cv.written[path] = self.k
        return W()


def test_open_retries_after_a_stall():
    cv, waits, logs = FakeCV([(0, 0), (0, 0), (9000, 9000)]), [], []
    cap, fps, n_all, tries = open_clip("u", cv, tries=3, wait_s=20, sleep=waits.append, log=logs.append)
    assert cap is not None and n_all == 9000 and tries == 3 and fps == 25.0
    assert waits == [20, 40] and len(logs) == 2


def test_open_gives_up_after_all_tries():
    cv, waits = FakeCV([(0, 0)]), []
    cap, fps, n_all, tries = open_clip("u", cv, tries=3, sleep=waits.append, log=lambda m: None)
    assert cap is None and n_all == 0 and cv.opens == 3 and len(waits) == 2


def test_open_first_try_no_wait():
    cv, waits = FakeCV([(9000, 9000)]), []
    cap, _, _, tries = open_clip("u", cv, sleep=waits.append, log=lambda m: None)
    assert tries == 1 and waits == []


def test_piece_reread_when_stream_dies_part_way(tmp_path):
    out = str(tmp_path / "piece.mp4")
    cv, waits = FakeCV([(9000, 120), (9000, 500)]), []
    cap, _, _, _ = open_clip("u", cv, log=lambda m: None)
    got, fps, tries = read_piece("u", cv, 100, 500, out, sleep=waits.append, log=lambda m: None, cap=cap)
    assert got == 500 and tries == 2 and waits == [20]
    assert len(open(out).read()) == 500 and not [p for p in os.listdir(tmp_path) if ".try" in p]


def test_piece_keeps_longest_when_all_short(tmp_path):
    out = str(tmp_path / "piece.mp4")
    cv = FakeCV([(9000, 300), (0, 0), (9000, 100)])
    got, _, tries = read_piece("u", cv, 0, 500, out, sleep=lambda s: None, log=lambda m: None)
    assert got == 300 and tries == 3 and len(open(out).read()) == 300
    assert not [p for p in os.listdir(tmp_path) if ".try" in p]


def test_piece_unreadable_returns_zero(tmp_path):
    out = str(tmp_path / "piece.mp4")
    got, _, _ = read_piece("u", FakeCV([(0, 0)]), 0, 500, out, sleep=lambda s: None, log=lambda m: None)
    assert got == 0 and not os.path.exists(out)


def test_tracktest_uses_retrying_open():
    s = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "tracktest.py")).read()
    assert "open_clip(src, cv2" in s and "read_piece(src, cv2" in s
    assert "-c user.email=bot@ipanema pull" in s        # progress push: the rebase needs an identity too (28 Oct log)
