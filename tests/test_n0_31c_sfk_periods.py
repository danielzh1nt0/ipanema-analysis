import os, sys, json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import n0_31c_sfk_periods as c

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_windows_cover_each_unchecked_boundary():
    per = json.load(open(os.path.join(ROOT, "periods", "SFKBP1109.json")))["periods_s"]
    ts = [t for _, t in c.window_times()]
    for b in (per[0][1], per[1][0], per[1][1]):       # first-half end, second-half kick-off, full time
        assert any(abs(t - b) <= 5 for t in ts), b
    assert ts == sorted(ts) and len(ts) == len(set(ts))


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
