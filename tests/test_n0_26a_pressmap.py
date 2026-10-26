"""N0-26a (26 Oct): the app's pressing-over-time chart / pressure map (metrics.pressure_points) checked on Metrica pro data
by tools/n0_26a_pressmap.py; option IPANEMA_PRESS_POINTS=spell counts only while the spell state gives the carrier's team
the ball (better on Metrica, worse by eye on our clips -> off by default)."""
import os, sys, importlib, subprocess
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from ipanema import metrics as M

FPS = 25


def _frames(n, team="A", carrier=5, pm=1.5):
    return [{"carrier": carrier, "pressure_m": pm, "team": team, "pos": [50.0, 30.0]} for _ in range(n)]


def test_state_gate_keeps_only_frames_where_the_carriers_team_has_the_ball():
    fr = _frames(100)
    state = [0] * 25 + [2] * 25 + [1] * 25 + [0] * 25            # A, loose, B, A
    out = M.pressure_points(None, fr, FPS, state=state)
    assert [p["t"] for p in out] == [0.0, 3.0]                      # loose and B's ball give no point to A's carrier
    assert [p["t"] for p in M.pressure_points(None, fr, FPS)] == [0.0, 1.0, 2.0, 3.0]   # old rule: every second


def test_state_gate_dead_ball_and_short_state():
    fr = _frames(50, team="B")
    assert M.pressure_points(None, fr, FPS, state=[3] * 50) == []   # dead ball: nobody presses
    out = M.pressure_points(None, fr, FPS, state=[1] * 30)          # state shorter than the frames: the tail is skipped
    assert [p["t"] for p in out] == [0.0, 1.0] and out[0]["pressing_team"] == "A"


def test_compute_default_raw_and_env_spell(monkeypatch):
    n = 100; state = np.array([0] * 50 + [2] * 50); ballm = {}; fr = _frames(n)
    args = (state, ballm, {}, FPS, 105.0, 68.0, {"A": True, "B": False}, [], [], [], [], [{} for _ in range(n)])
    assert len(M.compute(*args, frames_=fr, cstate=state)["pressure_points"]) == 4      # default: the old raw rule
    monkeypatch.setenv("IPANEMA_PRESS_POINTS", "spell"); M2 = importlib.reload(M)
    try: assert [p["t"] for p in M2.compute(*args, frames_=fr, cstate=state)["pressure_points"]] == [0.0, 1.0]
    finally:
        monkeypatch.delenv("IPANEMA_PRESS_POINTS"); importlib.reload(M)


def test_run_py_passes_the_spell_state():
    src = open(os.path.join(ROOT, "ipanema", "run.py")).read()
    assert "frames_=frames_, cstate=cstate)" in src


def test_truth_points_skip_dead_ball_and_one_per_second():
    import n0_26a_pressmap as T
    ks = list(range(100))
    fr = {k: (1, [(105, 50.0, 30.0), (207, 51.0, 30.0), (208, 70.0, 30.0)], (50.0, 30.0) if k < 60 else None) for k in ks}
    car = {k: 105 for k in ks}
    pts = T.truth_points(fr, 1, car, ks)
    assert [p["t"] for p in pts] == [0.0, 1.0, 2.0] and all(p["pressing_team"] == "B" for p in pts)   # ball out from 60


def test_match_points_one_to_one_same_team_time_and_place():
    import n0_26a_pressmap as T
    tr = [{"t": 10.0, "pressing_team": "A", "x_m": 50.0, "y_m": 30.0}, {"t": 20.0, "pressing_team": "B", "x_m": 50.0, "y_m": 30.0}]
    ours = [{"t": 10.5, "pressing_team": "A", "x_m": 52.0, "y_m": 31.0}, {"t": 10.6, "pressing_team": "A", "x_m": 51.0, "y_m": 30.0},   # 2nd: truth already used
            {"t": 20.0, "pressing_team": "A", "x_m": 50.0, "y_m": 30.0},                                                           # wrong team
            {"t": 30.0, "pressing_team": "B", "x_m": 50.0, "y_m": 30.0}]                                                           # no truth near
    assert T.match_points(ours, tr) == 1


def test_grade_blocks_and_thirds():
    import n0_26a_pressmap as T
    ar = {"A": True, "B": False}
    tr = [{"t": float(i), "pressing_team": "A", "x_m": 90.0, "y_m": 30.0} for i in range(10)] + [{"t": 310.0, "pressing_team": "B", "x_m": 90.0, "y_m": 30.0}]
    g = T.grade(tr, tr, ar, ar, 600 * FPS)
    assert g["ours_real_pct"] == 100.0 and g["teams"]["A"]["ratio"] == 1.0 and g["teams"]["A"]["thirds_truth"] == [0, 0, 100.0]
    assert g["teams"]["B"]["thirds_truth"] == [100.0, 0, 0]        # x 90 is B's own third (B attacks left)
    assert g["block_top_right"] == g["block_top_n"] == 2 and g["more_pressing_right"]
