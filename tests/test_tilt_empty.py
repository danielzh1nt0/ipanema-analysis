"""N0-22d (22 Oct): metrics.tilt_windows (the app's 'possession over time' chart) gave possession_A 0.0 - the opponent 100% -
for a 15-s window in which neither team had the ball (before kick-off, half-time, after the whistle; since N0-21d those
frames are nobody's). Now None, plus control_s; IPANEMA_TILT_EMPTY=zero = old."""
import os, json, importlib, tempfile
import numpy as np
from ipanema import metrics as M

FPS, L = 10, 106.0
AR = {"A": True, "B": False}


def _reload(monkeypatch, val):
    if val is None: monkeypatch.delenv("IPANEMA_TILT_EMPTY", raising=False)
    else: monkeypatch.setenv("IPANEMA_TILT_EMPTY", val)
    return importlib.reload(M)


def test_empty_window_is_null_and_control_seconds(monkeypatch):
    m = _reload(monkeypatch, None)
    state = [0] * 150 + [2] * 150 + [1] * 30 + [0] * 120          # A 15 s | nobody 15 s | B 3 s + A 12 s
    ballm = {k: np.array([60.0, 30.0]) for k in range(450)}
    out = m.tilt_windows(state, ballm, FPS, L, AR, window_s=15)
    assert [w["possession_A"] for w in out] == [1.0, None, 0.8]
    assert [w["control_s"] for w in out] == [15.0, 0.0, 15.0]


def test_window_without_ball_metres_is_null(monkeypatch):
    m = _reload(monkeypatch, None)
    state = [1] * 300
    ballm = {k: np.array([60.0, 30.0]) for k in range(150)}       # no ball position in the second window
    out = m.tilt_windows(state, ballm, FPS, L, AR, window_s=15)
    assert [(w["possession_A"], w["control_s"]) for w in out] == [(0.0, 15.0), (None, 0.0)]   # a real 0.0 stays 0.0


def test_old_switch(monkeypatch):
    m = _reload(monkeypatch, "zero")
    out = m.tilt_windows([2] * 150, {k: np.array([60.0, 30.0]) for k in range(150)}, FPS, L, AR, window_s=15)
    assert out[0]["possession_A"] == 0.0
    _reload(monkeypatch, None)


def test_full_match_break_window_is_null_through_analyse(monkeypatch):
    """the 40-s full match of test_play_state with a longer break (15-35 s): the 15-30 s window lies in the break"""
    _reload(monkeypatch, None)
    from tests.test_play_state import _match
    from ipanema import run as R
    from ipanema.config import Settings
    monkeypatch.setattr(R, "_PLAY_STATE", True); monkeypatch.setattr(R, "_ball_gt", lambda *a, **k: None)
    root = tempfile.mkdtemp(); os.makedirs(f"{root}/cache/PLAYTEST", exist_ok=True)
    S = Settings(root=root, sports_dir="/tmp/nosports", work=f"{root}/work")
    ctx = _match(root, dur_s=50.0, periods_s=((0, 15), (35, 50)))
    _, folder, _ = R.analyse(ctx, S, log=lambda *a: None, export_kw={"frame_stride": 1, "copy_video": False, "make_zip": False}, gt_path="/nonexistent")
    tw = json.load(open(f"{folder}/stats.json"))["metrics"]["tilt_windows"]
    brk = [w for w in tw if w["t"] >= 15 and w["t_end"] <= 35]
    assert brk and all(w["possession_A"] is None and w["control_s"] == 0.0 for w in brk), tw
    play = [w for w in tw if w["t_end"] <= 15]
    assert play and all(w["possession_A"] is not None and w["control_s"] > 0 for w in play), tw
