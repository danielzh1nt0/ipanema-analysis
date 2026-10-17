"""17 Oct (worker, N0): tests for the main session's F1 tools/poss_sheet.py (picture sheet of the 'who has the ball'
moments a Kaggle export gets wrong). Made-up 20-s video + export + key; the tool runs in a temp folder like on the runner."""
import gzip, json, os, subprocess, sys
import cv2, numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = "fake-match"


@pytest.fixture
def setup(tmp_path):
    os.makedirs(tmp_path / "r2" / M); vw = cv2.VideoWriter(str(tmp_path / "r2" / M / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
    for k in range(200): vw.write(np.full((360, 640, 3), (40, 120 + k % 50, 40), np.uint8))
    vw.release()
    frames = [{"t": round(i / 3, 3), "possession": "A" if i / 3 < 10 else "B", "phase": "play",
               "players": [{"id": 1, "team": "A", "px": [500, 600]}, {"id": 2, "team": "B", "px": [900, 600], "state": "filled"}, {"id": 3, "team": None, "px": [1300, 500]}],
               "ball": {"px": [520, 640], "state": "seen"} if i % 2 == 0 else None} for i in range(60)]   # 0 .. 19.67 s at 3/s
    os.makedirs(tmp_path / "exp" / "job"); json.dump({"frames": frames}, gzip.open(tmp_path / "exp" / "job" / "frames_lite.json.gz", "wt"))
    os.makedirs(tmp_path / "reference" / M)
    json.dump({"items": [{"id": "k1", "t": 5.0, "owner": "A"},      # right
                         {"id": "k2", "t": 12.0, "owner": "A"},     # wrong (ours B)
                         {"id": "k3", "t": 15.1, "owner": "none"},  # not a clear owner: ignored
                         {"id": "k4", "t": 300.0, "owner": "A"}]},  # outside the export: skipped, not compared with the last frame
              open(tmp_path / "reference" / M / "ball_key_graded.json", "w"))
    json.dump({"items": [{"id": "s1", "t": 3.4, "owner": "B"}]}, open(tmp_path / "reference" / M / "poss_sample.json", "w"))   # wrong (ours A)
    return tmp_path


def _run(tmp, *extra):
    env = dict(os.environ, R2_PUBLIC_URL=str(tmp / "r2"))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "poss_sheet.py"), str(tmp / "exp" / "job"), M, *extra], cwd=tmp, env=env, capture_output=True, text=True, timeout=120)
    return r, tmp / "results" / "qa" / "poss" / "job"


def test_only_misses_by_default(setup):
    r, out = _run(setup)
    assert r.returncode == 0, r.stderr
    rows = json.load(open(out / "index.json"))
    assert [(x["id"], x["eye"], x["ours"]) for x in rows] == [("s1", "B", "A"), ("k2", "A", "B")]   # sorted by time, both key files read
    assert "1 key moments outside the export skipped" in r.stdout
    img = cv2.imread(str(out / "k2.jpg")); assert img.shape == (800, 2222, 3)                      # frame 1422 wide + 800 zoom
    assert img[:50, :400].mean() > 5 and img[60:, :1422].mean() > 20                                # label bar written, picture not black


def test_all_includes_right_ones(setup):
    r, out = _run(setup, "all")
    assert r.returncode == 0, r.stderr
    assert [x["id"] for x in json.load(open(out / "index.json"))] == ["s1", "k1", "k2"]
    assert not (out / "k3.jpg").exists() and not (out / "k4.jpg").exists()


def test_empty_export_stops_cleanly(setup):
    json.dump({"frames": []}, gzip.open(setup / "exp" / "job" / "frames_lite.json.gz", "wt"))
    r, _ = _run(setup)
    assert r.returncode != 0 and "no frames" in r.stderr
