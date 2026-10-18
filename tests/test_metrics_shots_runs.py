"""N0 18 Oct: metrics.shots / high_turnovers / runs / pressure_points / tilt_windows had no tests (every app run calls them).
The runs test with a track gap fails on the old code (a sprint ending at a gap was thrown away)."""
import numpy as np
from ipanema import metrics as M

FPS, L, W = 10, 106.0, 64.0
AR = {"A": True, "B": False}           # A attacks x = L


def _shot_inputs(start=(80.0, 32.0), v=(2.0, 0.0), n=60, k0=20, team=0, dead_after=None):
    """ball still at `start` until k0, then moves v metres per frame toward a goal; team `team` (0 = A) has the ball"""
    state = [team] * n; ballm = {}; bspeed = {}
    p = np.array(start, float)
    for k in range(n):
        if k > k0: p = p + np.array(v)
        ballm[k] = p.copy(); bspeed[k] = float(np.linalg.norm(v)) * FPS if k > k0 else 0.0
    if dead_after is not None:
        for k in range(dead_after, n): state[k] = 3
    return state, ballm, bspeed


def test_shot_on_target_when_ball_crosses_the_goal_line():
    state, ballm, bspeed = _shot_inputs()               # 20 m/s from 26 m (24 m at the first fast frame) straight at the goal A attacks
    out = M.shots(state, ballm, bspeed, FPS, L, W, AR, [])
    assert len(out) == 1
    s = out[0]
    assert s["team"] == "A" and s["outcome"] == "on target" and not s["goal"] and s["speed_ms"] == 20.0 and s["distance_m"] == 24.0


def test_no_shot_when_slow_far_or_wrong_way():
    st, bm, bs = _shot_inputs(v=(1.0, 0.0))             # 10 m/s: not a shot
    assert M.shots(st, bm, bs, FPS, L, W, AR, []) == []
    st, bm, bs = _shot_inputs(start=(60.0, 32.0), n=27, dead_after=24)   # 46 m out (and play stops before it gets near): too far
    assert M.shots(st, bm, bs, FPS, L, W, AR, []) == []
    st, bm, bs = _shot_inputs(v=(-2.0, 0.0), dead_after=40)   # A kicks toward its own goal
    assert M.shots(st, bm, bs, FPS, L, W, AR, []) == []
    st, bm, bs = _shot_inputs(v=(0.0, 2.0), dead_after=40)    # sideways
    assert M.shots(st, bm, bs, FPS, L, W, AR, []) == []


def test_shot_off_target_from_the_goal_kick_and_team_b_attacks_left():
    st, bm, bs = _shot_inputs(start=(80.0, 26.0), v=(2.0, 0.0), dead_after=32)   # wide; play stops
    gk = [{"t": 4.5, "kind": "goal kick", "team": "B", "x_m": 104.0, "y_m": 32.0}]
    out = M.shots(st, bm, bs, FPS, L, W, AR, gk)
    assert [s["outcome"] for s in out] == ["off target"]
    st, bm, bs = _shot_inputs(start=(26.0, 32.0), v=(-2.0, 0.0), team=1)            # B shoots at x = 0
    out = M.shots(st, bm, bs, FPS, L, W, AR, [])
    assert [(s["team"], s["outcome"]) for s in out] == [("B", "on target")]


def test_shot_that_does_not_end_play_is_not_counted_and_outcome_labels_are_consistent():
    st, bm, bs = _shot_inputs(v=(2.0, 0.0), n=24)       # ball fast but play neither crosses nor stops in the window
    assert M.shots(st, bm, bs, FPS, L, W, AR, []) == []
    st, bm, bs = _shot_inputs(start=(80.0, 26.0), v=(2.0, 0.0), dead_after=32)
    out = M.shots(st, bm, bs, FPS, L, W, AR, [])        # no restart known -> default label
    assert out[0]["outcome"] == "saved or blocked"
    corner = [{"t": 4.5, "kind": "corner", "team": "A", "x_m": 105.0, "y_m": 63.0}]
    assert M.shots(st, bm, bs, FPS, L, W, AR, corner)[0]["outcome"] == "saved or blocked"


def test_centre_spot_restart_is_not_a_goal_today():
    """pins the known dead rule (see the shots docstring): restarts at the centre are 'stoppage', so no goal from our detector"""
    st, bm, bs = _shot_inputs(start=(80.0, 26.0), v=(2.0, 0.0), dead_after=32)
    ko = [{"t": 15.0, "kind": "stoppage", "team": "B", "x_m": 53.0, "y_m": 32.0}]
    assert not M.shots(st, bm, bs, FPS, L, W, AR, ko)[0]["goal"]


def test_high_turnovers_zones_per_attack_direction():
    ballm = {10: np.array([90.0, 30.0]), 20: np.array([90.0, 30.0]), 30: np.array([50.0, 30.0])}
    tv = [{"t": 1.0, "frame": 10, "won_by": "A"}, {"t": 2.0, "frame": 20, "won_by": "B"}, {"t": 3.0, "frame": 30, "won_by": "A"}, {"t": 4.0, "frame": 99, "won_by": "A"}]
    out = M.high_turnovers(tv, ballm, L, AR, FPS)
    assert [(h["team"], h["zone"]) for h in out] == [("A", "high"), ("B", "low"), ("A", "mid")]   # frame 99 has no ball -> left out


def _track(xs, tid=7, team="A", start=0):
    return {start + i: [(tid, team, np.array([x, 30.0]))] for i, x in enumerate(xs)}


def test_runs_finds_a_sprint_and_ignores_jogging_and_id_jumps():
    per = _track([10 + 0.7 * i for i in range(15)])                       # 7 m/s for 1.4 s
    out = M.runs(per, FPS, [0] * 15, AR)
    assert len(out) == 1 and out[0]["player"] == 7 and out[0]["speed_ms"] == 7.0 and out[0]["with_ball"] is True
    assert M.runs(_track([10 + 0.3 * i for i in range(15)]), FPS, [0] * 15, AR) == []          # 3 m/s
    assert M.runs(_track([10 + 2.0 * i for i in range(15)]), FPS, [0] * 15, AR) == []          # 20 m/s = an ID swap, not a sprint


def test_runs_keeps_a_sprint_that_ends_at_a_track_gap():
    per = _track([10 + 0.7 * i for i in range(12)])                       # 1.1 s sprint ...
    per.update(_track([40.0, 40.1, 40.2], start=20))                      # ... then the track is lost for 0.9 s
    for k in range(12, 20): per.setdefault(k, [])
    out = M.runs(per, FPS, [1] * 23, AR)
    assert len(out) == 1 and out[0]["t"] == 0.0 and out[0]["t_end"] == 1.1 and out[0]["with_ball"] is False


def test_pressure_points_one_per_second_per_carrier():
    fr = [{"carrier": 5, "pressure_m": 1.5, "team": "A", "pos": [50.0, 30.0]} for _ in range(25)]
    fr[3]["pressure_m"] = 4.0
    out = M.pressure_points(None, fr, FPS)
    assert [p["t"] for p in out] == [0.0, 1.0, 2.0] and out[0]["pressing_team"] == "B"


def test_tilt_windows():
    state = [0] * 150 + [1] * 150
    ballm = {k: np.array([100.0, 30.0]) for k in range(300)}                # A in its final third, B deep in its own half
    out = M.tilt_windows(state, ballm, FPS, L, AR, window_s=15)
    assert [(w["tilt_A"], w["possession_A"]) for w in out] == [(1.0, 1.0), (None, 0.0)]
