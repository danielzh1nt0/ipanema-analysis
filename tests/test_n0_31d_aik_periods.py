import os, sys, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import n0_31d_aik_periods as d
import n0_31c_sfk_periods as c

ROOT = os.path.join(os.path.dirname(__file__), "..")
PER = os.path.join(ROOT, "periods", "p15u-vs-aik-2026-09-21-bd09.json")


def test_windows_cover_the_old_and_suspected_boundaries():
    ts = [t for _, t in c.window_times(d.WINDOWS)]
    for b in (366, 437, 3225, 3950, 3995, 6679, 6777):   # old file values + where earlier side findings put them
        assert any(abs(t - b) <= 5 for t in ts), b
    assert ts == sorted(set(ts))
    assert len(ts) < 120                                  # a quick free-runner job


def test_dry_run_on_a_local_stand_in_video(tmp_path, monkeypatch):
    import pytest
    cv2 = pytest.importorskip("cv2"); np = pytest.importorskip("numpy")
    v = tmp_path / "t.mp4"
    w = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (320, 180))
    for k in range(250):
        w.write(np.full((180, 320, 3), k % 250, np.uint8))
    w.release()
    monkeypatch.setenv("SRC", str(v)); monkeypatch.setenv("WIN", "a:1:9:4")
    monkeypatch.setattr(d, "OUT", str(tmp_path / "o"))
    d.main()
    meta = json.load(open(tmp_path / "o" / "times.json"))
    assert meta["match"] == d.M and [f["t"] for f in meta["frames"]] == [1.0, 5.0, 9.0]


def test_periods_file_holds_the_by_eye_values():
    p = json.load(open(PER))
    per = p["periods_s"]
    assert per == [[440, 3225], [3995, 6745]]            # N0-31d by eye (results/free/n0_31d)
    assert p["attack_right_A"] is True and p["trust_all_calibration"] is True
    assert "N0-31d" in p["note"]
