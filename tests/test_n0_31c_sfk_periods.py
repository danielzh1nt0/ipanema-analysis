import os, sys, json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import n0_31c_sfk_periods as c

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_windows_cover_each_unchecked_boundary():
    per = json.load(open(os.path.join(ROOT, "periods", "SFKBP1109.json")))["periods_s"]
    ts = [t for _, t in c.window_times()] + [t for _, t in c.window_times(c.parse_windows(c.WIN_DEFAULT))]
    for b in (per[0][1], per[1][0], per[1][1]):       # first-half end, second-half kick-off, full time
        assert any(abs(t - b) <= 5 for t in ts), b
    w1 = [t for _, t in c.window_times()]; assert w1 == sorted(set(w1))   # each run: times in order, no repeats


def test_periods_file_has_the_by_eye_second_half_kickoff():
    per = json.load(open(os.path.join(ROOT, "periods", "SFKBP1109.json")))["periods_s"]
    assert per[0] == [555, 3060]                      # first half: kick-off (2 Oct) and end (31 Oct) both by eye
    assert per[1][0] == 3510                          # was 3733 - play is on at 3620 (results/free/n0_31c/sheet_ko2.jpg)
    assert per[0][1] < per[1][0] < per[1][1]


def test_cut_on_a_local_stand_in_video(tmp_path):
    cv2 = pytest.importorskip("cv2"); np = pytest.importorskip("numpy")
    v = tmp_path / "t.mp4"
    w = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 180))
    for k in range(300):                                # 10 s, each frame a different grey so seeks are visible
        w.write(np.full((180, 320, 3), k % 250, np.uint8))
    w.release()
    meta = c.cut(str(v), [("a", 1.0), ("a", 5.0), ("b", 8.0), ("b", 99.0)], str(tmp_path / "o"), cv2, np,
                 log=lambda *a: None)
    assert [f["file"] for f in meta["frames"]] == ["a_00001.jpg", "a_00005.jpg", "b_00008.jpg"]   # 99 s past the end
    assert meta["video_s"] == 10.0
    assert (tmp_path / "o" / "sheet_a.jpg").exists() and (tmp_path / "o" / "sheet_b.jpg").exists()
    img = cv2.imread(str(tmp_path / "o" / "a_00005.jpg"))
    assert img.shape[1] == c.W
    assert abs(float(img[200:, :].mean()) - 150) < 5    # seek landed on frame 150 (below the time label)


def test_window_spec_for_the_second_run():
    w = c.parse_windows("ko2b:3290:3615:5")
    assert w == [("ko2b", 3290.0, 3615.0, 5.0)]
    ts = [t for _, t in c.window_times(w)]
    assert ts[0] == 3290.0 and ts[-1] == 3615.0 and len(ts) == 66
    assert c.parse_windows("") == c.WINDOWS
