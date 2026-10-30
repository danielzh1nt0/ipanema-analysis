"""N0-30d (30 Oct): the 'X m in 5 s' on every 'won the ball' event (payload.gain_5s_m) read the ball at exactly +5 s
wherever it projected - on the app copies -406 m and 923 m (pitch 106 m). Now both ends must be on the pitch (+2 m): start =
first such ball in the first second, end = the last such ball within the 5 s; under 1 s of usable ball -> None.
IPANEMA_GAIN_ONPITCH=0 = old. Metrica: same error (perfect ball 4.2 -> 3.9 m), a figure for 82 -> 95% of wins."""
import os, sys, json
import numpy as np
from ipanema import possession as P
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

FPS = 10; PITCH = (106.0, 64.0)


def _ball(fn, n=80):
    return {k: np.array(fn(k / FPS)) for k in range(n) if fn(k / FPS) is not None}


def test_on_pitch_same_as_old():
    b = _ball(lambda t: [40.0 + 3 * t, 30.0])                       # 3 m/s straight to the right
    assert P.gain_window(b, 0, 80, FPS, True, PITCH, onpitch=True) == P.gain_window(b, 0, 80, FPS, True, PITCH, onpitch=False) == 15.0
    assert P.gain_window(b, 0, 80, FPS, False, PITCH, onpitch=True) == -15.0   # winner attacks left


def test_wild_end_ignored_last_on_pitch_used():
    b = _ball(lambda t: [40.0 + 4 * t, 30.0] if t < 3.0 else [-104.0, 570.0])   # ball projects far off the pitch from 3 s
    assert P.gain_window(b, 0, 80, FPS, True, PITCH, onpitch=False) == -144.0   # old: absurd
    g, j0, j5 = P.gain_window(b, 0, 80, FPS, True, PITCH, onpitch=True, detail=True)
    assert (j0, j5) == (0, 29) and g == 11.6                       # where the ball was last seen on the pitch (2.9 s)


def test_ball_out_over_end_line_counts_to_the_line():
    b = _ball(lambda t: [90.0 + 5 * t, 30.0])                      # crosses x=106 at 3.2 s; +2 m margin, then clamped to L
    assert P.gain_window(b, 0, 80, FPS, True, PITCH, onpitch=True) == 16.0


def test_start_off_pitch_takes_first_on_pitch_ball_within_1s():
    b = _ball(lambda t: [-30.0, 30.0] if t < 0.5 else [50.0 + 2 * (t - 0.5), 30.0])
    g, j0, _ = P.gain_window(b, 0, 80, FPS, True, PITCH, onpitch=True, detail=True)
    assert j0 == 5 and g == 9.0
    b2 = _ball(lambda t: [-30.0, 30.0] if t < 1.5 else [50.0, 30.0])       # nothing usable in the first second
    assert P.gain_window(b2, 0, 80, FPS, True, PITCH, onpitch=True) is None


def test_under_one_second_of_usable_ball_is_none():
    b = _ball(lambda t: [50.0, 30.0] if t < 0.5 else None)
    assert P.gain_window(b, 0, 80, FPS, True, PITCH, onpitch=True) is None
    assert P.gain_window(b, 0, 80, FPS, True, PITCH, onpitch=False) is None


def test_no_pitch_is_old_rule():
    b = _ball(lambda t: [40.0 + 4 * t, 30.0] if t < 3.0 else [-104.0, 570.0])
    assert P.gain_window(b, 0, 80, FPS, True, None, onpitch=True) == -144.0


def test_turnovers_use_the_pitch(monkeypatch):
    """B holds 0-4.9 s, A from 5 s; ball flies off the pitch (wild projection) 8 s on"""
    n = 150; per = {}; frames_ = []; state = []; ballm = {}
    for k in range(n):
        t = k / FPS
        per[k] = [[1, "A", np.array([40.0, 30.0]), None, None, False], [2, "B", np.array([38.0, 32.0]), None, None, False]]
        if t < 5.0: c, tm, pos, st, b = 2, "B", [38.0, 32.0], 1, [38.0, 32.0]
        else: c, tm, pos, st, b = 1, "A", [40.0 + 3 * (t - 5), 30.0], 0, ([40.0 + 3 * (t - 5), 30.0] if t < 8.0 else [400.0, -300.0])
        frames_.append({"frame": k, "carrier": c, "team": tm, "pos": pos, "pressure_m": 5.0, "near_opps": 0})
        state.append(st); ballm[k] = np.array(b)
    monkeypatch.setattr(P, "GAIN_ONPITCH", True)
    tv = P.turnovers(per, frames_, state, ballm, FPS, {"A": True, "B": False}, pitch=PITCH)[0]
    assert tv["won_by"] == "A" and 7.0 <= tv["gain_5s_m"] <= 9.0          # ~2.9 s at 3 m/s before the wild spot
    monkeypatch.setattr(P, "GAIN_ONPITCH", False)
    tv = P.turnovers(per, frames_, state, ballm, FPS, {"A": True, "B": False}, pitch=PITCH)[0]
    assert tv["gain_5s_m"] > 300                                          # old: the wild spot


def test_lab_truth_uses_last_true_ball():
    import n0_30d_gain5s as G
    fr = {k: (1, [], (50.0 + 0.1 * k, 30.0) if k < 60 else None) for k in range(200)}   # true ball goes out at frame 60
    assert G.gain(fr, 0, True, last=False) is None
    assert G.gain(fr, 0, True) == 5.9
    pairs = [(10.0, 12.0), (-8.0, 3.0), (None, 1.0)]
    s = G.score(pairs)
    assert s["n"] == 2 and s["mae_m"] == 6.5 and s["dir_agree_pct"] == 50


def test_frame_picks_blind_and_reproducible(tmp_path):
    import n0_30d_frames as F
    rows = [{"fps": 30.0, "t_won": 10.0, "team": "A", "old": 5.0, "new": 20.0, "old_start": {"t": 10.0, "px": [1, 1]}, "new_start": {"t": 10.0, "px": [1, 1]},
             "old_end": {"t": 15.0, "px": [2, 2]}, "new_end": {"t": 13.0, "px": [3, 3]}},
            {"fps": 30.0, "t_won": 20.0, "team": "B", "old": 5.0, "new": 6.0, "old_start": {"t": 20.0, "px": [1, 1]}, "new_start": {"t": 20.0, "px": [1, 1]},
             "old_end": {"t": 25.0, "px": [2, 2]}, "new_end": {"t": 25.0, "px": [2, 2]}}]
    p = F.picks({"m": {"rows": rows}}, "m")
    assert len(p) == 1 and {p[0]["X"]["rule"], p[0]["Y"]["rule"]} == {"old", "new"}   # unchanged event left out
    assert p == F.picks({"m": {"rows": rows}}, "m")
