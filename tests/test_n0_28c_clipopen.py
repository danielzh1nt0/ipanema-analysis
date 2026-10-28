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


# --- the scorecard's copy step (N0-28c): each artifact holds every tracktest* folder of its checkout ---
def _artifact(root, match, folders):
    import json
    for fol, dark in folders.items():
        os.makedirs(os.path.join(root, f"players-{match}", fol), exist_ok=True)
        json.dump({"new, RF-DETR": {"observed_per_frame": {"dark": dark, "light": 1.0}, "median_track_s": {"dark": 1, "light": 1}}},
                  open(os.path.join(root, f"players-{match}", fol, "summary.json"), "w"))


def test_collect_takes_only_each_jobs_own_folder(tmp_path):
    import json
    from tools.nightly_scorecard import collect
    art, dest = str(tmp_path / "art"), str(tmp_path / "qa")
    # SFK job: fresh SFK (9) + stale repo copy of Solberga (1); Solberga job: fresh Solberga (7) + stale SFK (2)
    _artifact(art, "SFKBP1109_s1200", {"tracktest": 9.0, "tracktest_solberga": 1.0})
    _artifact(art, "solberga", {"tracktest": 2.0, "tracktest_solberga": 7.0})
    got = collect(art, dest)
    assert sorted(got) == ["tracktest", "tracktest_solberga"]
    rd = lambda f: json.load(open(os.path.join(dest, f, "summary.json")))["new, RF-DETR"]["observed_per_frame"]["dark"]
    assert rd("tracktest") == 9.0 and rd("tracktest_solberga") == 7.0        # old cp loop: 2.0 (stale) and 7.0


def test_collect_leaves_failed_jobs_folder_alone(tmp_path):
    from tools.nightly_scorecard import collect
    art, dest = str(tmp_path / "art"), str(tmp_path / "qa")
    os.makedirs(os.path.join(dest, "tracktest_djm")); open(os.path.join(dest, "tracktest_djm", "summary.json"), "w").write("{}")
    _artifact(art, "solberga", {"tracktest_djm": 3.0, "tracktest_solberga": 7.0})   # djm job failed: no players-djm
    assert collect(art, dest) == ["tracktest_solberga"]
    assert open(os.path.join(dest, "tracktest_djm", "summary.json")).read() == "{}"


def test_collect_noop_without_artifacts(tmp_path):
    from tools.nightly_scorecard import collect
    assert collect(str(tmp_path / "none"), str(tmp_path / "qa")) == []


def test_nightly_yml_uses_collect():
    s = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".github", "workflows", "nightly.yml")).read()
    assert "nightly_scorecard.py --collect" in s and "cp -r $d/*" not in s


def test_tracktest_fails_on_a_cut_piece():
    s = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "tracktest.py")).read()
    assert "_got < 0.9 * n" in s


class _R:
    def __init__(self, rc=0, out="", err=""): self.returncode, self.stdout, self.stderr = rc, out, err


def test_local_copy_downloads_small_clip(tmp_path):
    from tools.clipopen import local_copy
    dest = str(tmp_path / "c.mp4"); calls = []
    def run(cmd, **k):
        calls.append(cmd)
        if cmd[1].endswith("I"): return _R(0, "HTTP/1.1 200 OK\r\ncontent-length: 5\r\n")
        open(dest, "wb").write(b"12345"); return _R(0)
    assert local_copy("https://x/v.mp4", dest, max_bytes=100, run=run, log=lambda m: None) == dest and len(calls) == 2


def test_local_copy_streams_big_or_failed_or_local(tmp_path):
    from tools.clipopen import local_copy
    dest = str(tmp_path / "c.mp4")
    big = lambda cmd, **k: _R(0, "Content-Length: 4085890041\r\n")
    assert local_copy("https://x/v.mp4", dest, max_bytes=1.5e9, run=big, log=lambda m: None) == "https://x/v.mp4"
    def short(cmd, **k):
        if cmd[1].endswith("I"): return _R(0, "content-length: 9\r\n")
        open(dest, "wb").write(b"123"); return _R(18, err="partial")
    assert local_copy("https://x/v.mp4", dest, max_bytes=100, run=short, log=lambda m: None) == "https://x/v.mp4"
    assert local_copy("/data/v.mp4", dest, run=None, log=lambda m: None) == "/data/v.mp4"
