"""17 Oct (worker, N0): tests for the main session's T1 changes - IPANEMA_NEAR_M (possession nearness) and
tools/poss_sample_sheet.py (blind by-eye possession sheets)."""
import gzip, json, os, subprocess, sys
import numpy as np
import pytest
from ipanema import possession as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _two_players(n=30, ball_x=102.5):
    H = {k: np.eye(3) for k in range(n)}                                     # 1 px = 1 m side to side
    per = {k: [[1, "A", None, np.array([100.0, 100.0]), None, False], [2, "B", None, np.array([140.0, 100.0]), None, False]] for k in range(n)}
    ball = {k: (ball_x, 100.0) for k in range(n)}                            # 2.5 m from A
    return per, ball, H


def test_near_m_default_unchanged(monkeypatch):
    monkeypatch.delenv("IPANEMA_NEAR_M", raising=False)
    per, ball, H = _two_players()
    assert list(P.possession_simple(per, ball, H, 30, smooth=2)) == list(P.possession_simple(per, ball, H, 30, near_m=1.5, smooth=2))
    assert set(P.possession_simple(per, ball, H, 30, smooth=2)) == {2}       # 2.5 m > 1.5 m: loose


def test_near_m_env_and_explicit_wins(monkeypatch):
    per, ball, H = _two_players()
    monkeypatch.setenv("IPANEMA_NEAR_M", "3.0")
    assert set(P.possession_simple(per, ball, H, 30, smooth=2)) == {0}       # 2.5 m <= 3 m: A has it
    assert set(P.possession_simple(per, ball, H, 30, near_m=1.5, smooth=2)) == {2}   # an explicit value beats the env


def test_kaggle_template_sets_near_only_when_given():
    src = open(os.path.join(ROOT, "kaggle", "full_match.py")).read()
    assert 'if "{{NEAR}}": os.environ["IPANEMA_NEAR_M"] = "{{NEAR}}"' in src
    env = {}; exec('if "": env["X"] = ""', {"env": env}); assert env == {}   # left empty: the default stays
    exec('if "2.5": env["X"] = "2.5"', {"env": env}); assert env == {"X": "2.5"}


@pytest.fixture
def fake_match(tmp_path):
    cv2 = pytest.importorskip("cv2")
    M = "m_test"; vid = tmp_path / "r2" / M; vid.mkdir(parents=True)
    fps, secs = 10, 120
    w = cv2.VideoWriter(str(vid / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 360))
    for i in range(fps * secs):
        img = np.full((360, 640, 3), (40, 120, 40), np.uint8); cv2.putText(img, str(i // fps), (20, 60), 0, 1.5, (255, 255, 255), 3); w.write(img)
    w.release()
    src = tmp_path / "results" / "kaggle" / "run1"; src.mkdir(parents=True)
    frames = [{"t": round(k * 0.3, 2), "ball": {"px": [1800.0, 100.0]}, "possession": k % 3, "phase": "play"} for k in range(int(secs / 0.3))]
    with gzip.open(src / "frames_lite.json.gz", "wt") as f: json.dump({"frames": frames}, f)
    (tmp_path / "periods").mkdir(); json.dump({"periods_s": [[10, 50], [70, 110]]}, open(tmp_path / "periods" / f"{M}.json", "w"))
    return tmp_path, M


def _run(tmp, M, n):
    env = dict(os.environ, R2_PUBLIC_URL=str(tmp / "r2"), PYTHONPATH=ROOT)
    return subprocess.run([sys.executable, os.path.join(ROOT, "tools", "poss_sample_sheet.py"), "results/kaggle/run1", M, str(n), "3"],
                          cwd=tmp, env=env, capture_output=True, text=True, timeout=120)


def test_sheet_moments_in_play_and_apart(fake_match):
    cv2 = pytest.importorskip("cv2")
    tmp, M = fake_match; r = _run(tmp, M, 10)
    assert r.returncode == 0, r.stderr
    out = tmp / "results" / "qa" / "poss_sample" / "run1"; idx = json.load(open(out / "index.json"))
    ts = [x["t"] for x in idx]
    assert len(idx) == 10 and ts == sorted(ts)
    assert all(10 <= t <= 50 or 70 <= t <= 110 for t in ts)                  # only inside the periods
    assert min(np.diff(ts)) > 5                                              # moments at least 5 s apart
    sheets = sorted(out.glob("sheet_*.jpg")); assert len(sheets) == 1        # up to 12 per sheet
    a = cv2.imread(str(sheets[0])); assert a.shape == (4 * 320, 3 * 560, 3)  # rows of 3, last row (1 crop) padded
    assert a[3 * 320:, 560:].max() < 30                                      # the padding is black


def test_sheet_short_match_stops(fake_match):
    tmp, M = fake_match; r = _run(tmp, M, 40)                                # 80 s of play can't hold 40 moments 5 s apart (before the guard: endless loop; 14 already hangs with seed 3)
    assert r.returncode == 0, r.stderr
    idx = json.load(open(tmp / "results" / "qa" / "poss_sample" / "run1" / "index.json"))
    assert 8 <= len(idx) < 40
