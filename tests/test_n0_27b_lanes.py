"""N0-27b (27 Oct): pass quality (passes[].lane_open / quality, players[].pass_quality, events pass_bad / pass_risky).
A lost pass's 'to' is the opponent who won it, so it never had a lane and bad_lost / execution_error stayed ~0. Now the
intended receiver of a lost pass = the passer's team-mate in reach nearest where the ball was won (passes[].intended_to);
IPANEMA_LOST_LANE=old = old behaviour. Completed passes and 'a better pass was open' are unchanged."""
import os, sys, subprocess, numpy as np
from ipanema import analytics as AN

FPS = 25.0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rows(spots):
    return [[t, "A" if t < 200 else "B", np.array(xy, float), None, None, False] for t, xy in spots.items()]


# 101 passes towards 103 (forward); 201 wins it at 103's spot area. 202 stands on the line 101 -> 103 or not.
def _setup(blocker_on_line):
    sp = {101: (30.0, 30.0), 102: (30.0, 50.0), 103: (55.0, 30.0), 201: (56.0, 31.0), 202: (42.0, 30.5 if blocker_on_line else 45.0)}
    n = 60; per = {k: _rows(sp) for k in range(n)}
    frames_ = [{"carrier": 101 if k < 25 else (201 if k >= 35 else None), "team": "A" if k < 25 else ("B" if k >= 35 else None),
                "pressure_m": None, "near_opps": 0, "pos": None} for k in range(n)]
    return per, frames_


def _passes(blocker_on_line):
    per, frames_ = _setup(blocker_on_line); ar = {"A": True, "B": False}
    return AN.passes(per, frames_, [], AN.lanes(per, frames_, ar), ar, FPS, min_pass_m=0)[0]


def test_lost_pass_through_blocked_lane_is_bad_lost():
    ps = _passes(True)
    assert len(ps) == 1 and not ps[0]["completed"] and ps[0]["to"] == 201
    assert ps[0]["intended_to"] == 103 and ps[0]["lane_open"] is False and ps[0]["quality"] == "bad_lost"


def test_lost_pass_through_open_lane_is_execution_error():
    ps = _passes(False)
    assert ps[0]["intended_to"] == 103 and ps[0]["lane_open"] is True and ps[0]["quality"] == "execution_error"
    assert ps[0]["better_option"] is False and "best_to" not in ps[0]                  # lost passes get no 'better option' (as before)


def test_old_switch(monkeypatch):
    monkeypatch.setattr(AN, "LOST_LANE", "old")
    ps = _passes(True)
    assert "intended_to" not in ps[0] and ps[0]["quality"] == "unknown" and ps[0]["lane_open"] is None


def test_intended_mate_nearest_to_spot():
    rows = _rows({101: (0.0, 0.0), 102: (10.0, 0.0), 103: (20.0, 5.0), 201: (19.0, 4.0)})
    ln = [{"to": 102, "open": True, "forward": True}, {"to": 103, "open": False, "forward": True}]
    assert AN.intended_mate(rows, ln, np.array([19.0, 4.0])) == 103
    assert AN.intended_mate(rows, [], np.array([19.0, 4.0])) is None


def test_completed_pass_unchanged(monkeypatch):
    sp = {101: (30.0, 30.0), 102: (35.0, 30.0), 103: (55.0, 40.0), 201: (45.0, 20.0)}
    per = {k: _rows(sp) for k in range(60)}
    frames_ = [{"carrier": 101 if k < 25 else (102 if k >= 35 else None), "team": "A" if (k < 25 or k >= 35) else None, "pressure_m": None, "near_opps": 0, "pos": None} for k in range(60)]
    ar = {"A": True, "B": False}; ln = AN.lanes(per, frames_, ar)
    new = AN.passes(per, frames_, [], ln, ar, FPS, min_pass_m=0)[0]
    monkeypatch.setattr(AN, "LOST_LANE", "old")
    old = AN.passes(per, frames_, [], ln, ar, FPS, min_pass_m=0)[0]
    assert new == old and "intended_to" not in new[0]


def test_metrica_tool_dry():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "n0_27b_lanes.py"), "--dry"], capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "PYTHONPATH": ROOT})
    assert r.returncode == 0 and '"ok": true' in r.stdout, r.stderr
