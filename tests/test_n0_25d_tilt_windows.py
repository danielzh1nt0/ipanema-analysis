"""N0-25d (25 Oct): the app's 'possession over time' / 'field tilt over time' (stats.metrics.tilt_windows: 15-s windows,
drawn by the app as a 5-min rolling mean) checked on Metrica pro data for the first time (tools/n0_25d_tilt_windows.py,
results/qa/n0-25d). No pipeline change; these tests pin what the chart rests on and the lab's truth rules."""
import os, sys
import numpy as np

from ipanema import metrics as ME

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import n0_25d_tilt_windows as T                                   # noqa: E402

FPS, L = 10.0, 105.0
AR = {"A": True, "B": False}


def test_windows_pool_frames_and_report_control():
    """possession_A = A's share of the CONTROLLED frames in the window (loose frames left out), control_s = seconds of
    control; tilt_A = A's share of the frames controlled in a final third"""
    n = 150                                                        # one 15-s window at 10 fps
    state = [0] * 60 + [2] * 30 + [1] * 60                         # A 6 s, loose 3 s, B 6 s
    ballm = {k: np.array([90.0, 34.0]) for k in range(n)}          # ball in A's final third (= B's own third)
    w = ME.tilt_windows(state, ballm, FPS, L, AR)
    assert len(w) == 1 and w[0]["possession_A"] == 0.5 and w[0]["control_s"] == 12.0
    assert w[0]["tilt_A"] == 1.0                                   # only A was in its final third


def test_empty_window_is_null_not_zero():
    w = ME.tilt_windows([2] * 150, {}, FPS, L, AR)
    assert w[0]["possession_A"] is None and w[0]["tilt_A"] is None and w[0]["control_s"] == 0.0


def _ws(vals, ctrl):
    return [{"t": 15.0 * i, "t_end": 15.0 * (i + 1), "possession_A": v, "tilt_A": None, "control_s": c} for i, (v, c) in enumerate(zip(vals, ctrl))]


def test_app_rolling_counts_every_window_the_same_and_lab_weights():
    """the app's 5-min curve averages windows (a 1-s window counts like a 15-s one); the lab weights by control_s"""
    ws = _ws([1.0, 0.0, None], [1.0, 15.0, 0.0])
    assert abs(T.rolling(ws, "possession_A")[0] - 0.5) < 1e-9
    assert abs(T.rolling(ws, "possession_A", weighted=True)[0] - 1 / 16) < 1e-9
    assert all(v is None for v in [w["possession_A"] for w in T.thin_null(ws, 2)][:1])


def test_bars_and_pooled_truth():
    ws = _ws([0.8] * 10 + [0.2] * 10 + [0.6] * 20, [10.0] * 40)  # 40 windows = 10 min -> two 5-min bars
    b = T.bars(ws, "possession_A"); p = T.pooled_bars(ws)
    assert len(b) == 2 and abs(b[0] - 0.5) < 1e-9 and abs(b[1] - 0.6) < 1e-9
    assert np.allclose(b, p)                                       # equal control -> mean of windows = pooled frames


def test_evaluate_identical_is_perfect():
    rng = np.random.default_rng(0)
    ws = _ws(list(rng.uniform(0.2, 0.8, 60)), list(rng.uniform(5, 15, 60)))
    r = T.evaluate(ws, ws)
    assert r["poss_window"]["med_pts"] == 0.0 and r["poss_bars_weighted"]["p90_pts"] < 1e-6 and r["poss_bars_app"]["p90_pts"] > 0  # unweighted bars differ from pooled frames when control varies
    assert r["poss_roll_weighted"]["side_right"] == r["poss_roll_weighted"]["clear"]


def test_truth_foul_stops_play_until_the_free_kick():
    """the truth credited a 2.7-min injury stoppage after a foul to the team of the last on-ball event; a foul is dead time"""
    ev = [{"Period": "1", "Type": "PASS", "Subtype": "", "Team": "Away", "Start Frame": "1", "End Frame": "5"},
          {"Period": "1", "Type": "CHALLENGE", "Subtype": "TACKLE-FAULT-LOST", "Team": "Away", "Start Frame": "10", "End Frame": "10"},
          {"Period": "1", "Type": "FAULT RECEIVED", "Subtype": "", "Team": "Home", "Start Frame": "11", "End Frame": "11"},
          {"Period": "1", "Type": "SET PIECE", "Subtype": "FREE KICK", "Team": "Home", "Start Frame": "50", "End Frame": "50"},
          {"Period": "1", "Type": "PASS", "Subtype": "", "Team": "Home", "Start Frame": "50", "End Frame": "60"}]
    fr = {k: (1, [], (50.0, 30.0)) for k in range(1, 80)}
    own = T.truth_own(ev, fr, 1)
    assert own[5] == "B" and 10 not in own and 30 not in own and own[55] == "A"
    assert T.is_foul(ev[1]) and T.is_foul(ev[2]) and not T.is_foul(ev[0])
