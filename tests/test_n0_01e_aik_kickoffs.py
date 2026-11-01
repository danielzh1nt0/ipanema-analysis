import os, sys, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import n0_01e_aik_kickoffs as e
import n0_31c_sfk_periods as c

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_goals_match_the_veo_list():
    goals = []
    for line in open(os.path.join(ROOT, "reference", f"veo_highlights_{e.M}.txt")):
        p = line.split()
        if len(p) >= 2 and p[1] == "goal":
            goals.append(int(float(p[0])))
    assert sorted(set(goals)) == e.GOALS


def test_windows_cover_every_kick_off_after_a_goal():
    ts = c.window_times(e.WINDOWS)
    for g in e.GOALS:
        mine = [t for n, t in ts if n == f"ko{g}"]
        assert min(mine) <= g + 30 and max(mine) >= g + 85          # 3128: kick-off ~+37 s, 2762: ~+61 s (N0-26b)
    assert len(ts) < 120                                            # a quick free-runner job


def test_dry_run_on_a_local_stand_in_video(tmp_path, monkeypatch):
    import pytest
    cv2 = pytest.importorskip("cv2"); np = pytest.importorskip("numpy")
    v = tmp_path / "t.mp4"
    w = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (640, 360))
    for k in range(250):
        w.write(np.full((360, 640, 3), k % 250, np.uint8))
    w.release()
    monkeypatch.setenv("SRC", str(v)); monkeypatch.setenv("WIN", "a:1:9:4")
    monkeypatch.setattr(e, "OUT", str(tmp_path / "o"))
    e.main()
    meta = json.load(open(tmp_path / "o" / "times.json"))
    assert meta["match"] == e.M and [f["t"] for f in meta["frames"]] == [1.0, 5.0, 9.0]
    assert cv2.imread(str(tmp_path / "o" / "a_00005.jpg")).shape[1] == e.WIDTH
