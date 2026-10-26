"""N0-26e (26 Oct): our shot detector (ipanema/metrics.shots). Goal = the first restart within 90 s is taken from the centre
spot (was: only kind 'unknown', which a restart with a ball spot never gets, so no shot was ever a goal);
IPANEMA_SHOT_END=stops (option): the restart step's stoppages end a shot too (possession_simple has no 'dead' state);
veo.score_detector matches from 2 s before to 30 s after a Veo clip start."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import metrics as M, veo as VEO

FPS, L, W = 25.0, 105.0, 68.0
AR = {"A": True, "B": False}


def _shot(cross=True, n=4000, k0=1000):
    """team A on the ball, then a 25 m/s ball from 20 m straight at the right goal; cross=False: it stops 3 m short"""
    state = np.full(n, 2); state[k0 - 20:k0 + 1] = 0
    ballm, bspeed = {}, {}
    for k in range(k0 - 20, k0 + 60):
        x = 85.0 + max(0, k - k0) * 1.0
        if not cross: x = min(x, 102.0)
        ballm[k] = np.array([x, 34.0]); bspeed[k] = 25.0 if k0 <= k < k0 + 15 and x < 102.0 else 0.0
    return state, ballm, bspeed


def test_goal_from_centre_restart():
    st, bm, bs = _shot()
    rst = [{"t": 1000 / FPS + 65, "kind": "stoppage", "x_m": 52.0, "y_m": 34.5, "team": "B"}]   # kick-off 65 s later
    new = M.shots(st, bm, bs, FPS, L, W, AR, rst, goal_rule="centre", end="state")
    old = M.shots(st, bm, bs, FPS, L, W, AR, rst, goal_rule="unknown", end="state")
    assert len(new) == 1 and new[0]["goal"] and new[0]["outcome"] == "goal" and new[0]["team"] == "A"
    assert len(old) == 1 and not old[0]["goal"] and old[0]["outcome"] == "on target"


def test_no_goal_when_next_restart_elsewhere():
    st, bm, bs = _shot()
    rst = [{"t": 1000 / FPS + 20, "kind": "goal kick", "x_m": 100.0, "y_m": 34.0, "team": "B"},
           {"t": 1000 / FPS + 70, "kind": "stoppage", "x_m": 52.5, "y_m": 34.0, "team": "B"}]
    s = M.shots(st, bm, bs, FPS, L, W, AR, rst, goal_rule="centre", end="state")
    assert len(s) == 1 and not s[0]["goal"]


def test_centre_restart_too_late_is_not_a_goal():
    st, bm, bs = _shot()
    rst = [{"t": 1000 / FPS + 120, "kind": "stoppage", "x_m": 52.5, "y_m": 34.0, "team": "B"}]
    assert not M.shots(st, bm, bs, FPS, L, W, AR, rst, goal_rule="centre", end="state")[0]["goal"]


def test_default_goal_rule_is_centre():
    assert M.SHOT_GOAL == "centre" or os.environ.get("IPANEMA_SHOT_GOAL")
    assert M.SHOT_END == "state" or os.environ.get("IPANEMA_SHOT_END")


def test_saved_shot_needs_stops_option():
    """possession_simple never says 'dead': a shot that stops short of the line is only found when the stoppages count"""
    st, bm, bs = _shot(cross=False)
    stops = [(1000 + 30, 1000 + 200)]                  # play stops 1.2 s after the shot
    assert M.shots(st, bm, bs, FPS, L, W, AR, [], stops=stops, end="state") == []
    s = M.shots(st, bm, bs, FPS, L, W, AR, [], stops=stops, end="stops")
    assert len(s) == 1 and s[0]["outcome"] == "saved or blocked"
    # a stoppage already running before the shot (ball kicked during a dead spell) does not end it
    assert M.shots(st, bm, bs, FPS, L, W, AR, [], stops=[(1000 - 100, 1000 + 200)], end="stops") == []
    # no stops handed in -> the state rule, whatever end says
    assert M.shots(st, bm, bs, FPS, L, W, AR, [], stops=None, end="stops") == []


def test_compute_passes_stops():
    st, bm, bs = _shot(cross=False); seen = {}
    orig = M.shots
    def spy(*a, **k): seen.update(k); return []
    M.shots = spy
    try:
        try: M.compute(st, bm, bs, FPS, L, W, AR, [], [], [], [], {k: {"A": None, "B": None} for k in range(len(st))}, stops=[(1, 2)])
        except Exception: pass                       # the rest of compute is not under test here
    finally: M.shots = orig
    assert seen.get("stops") == [(1, 2)]


def test_score_detector_window():
    veo = [{"t": 600.0}, {"t": 900.0}]
    r = VEO.score_detector([{"t": 620.0}, {"t": 895.0}, {"t": 1500.0}], veo)
    assert r == {"veo_shots": 2, "found": 1, "ours": 3, "real": 1}   # 620 = 20 s into the 600 clip; 895 is before the 900 clip
    assert VEO.score_detector([{"t": 598.5}], veo)["real"] == 1


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
