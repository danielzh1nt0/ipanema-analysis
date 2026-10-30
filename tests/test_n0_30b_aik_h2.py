import os, sys, json, subprocess
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import n0_30b_aik_h2_cut as c


def test_times_are_second_half_after_kickoff_and_fixed():
    ts = c.sample_times()
    assert len(ts) == 90 and ts == sorted(ts)
    assert all(3995.0 <= t <= 6899.0 for t in ts)
    assert ts == c.sample_times()                     # same seed -> same frames (blind sample is reproducible)


def test_cut_on_a_local_stand_in_video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    v = tmp_path / "t.mp4"
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30", "-t", "6", str(v)])
    if r.returncode != 0:
        pytest.skip("no ffmpeg")
    meta = c.cut(str(v), [1.0, 2.5, 99.0], str(tmp_path / "out"), cv2, log=lambda *a: None)
    assert [f["file"] for f in meta["frames"]] == ["q00.jpg", "q01.jpg"]  # 99 s is past the end -> skipped
    assert meta["frames"][1]["frame"] == 75
    assert json.load(open(tmp_path / "out" / "times.json"))["seed"] == 11
    assert (tmp_path / "out" / "q00.jpg").stat().st_size > 0
