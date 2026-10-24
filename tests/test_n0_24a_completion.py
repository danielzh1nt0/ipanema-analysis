"""N0-24a (24 Oct): the app's pass completion % (stats.json teams[].pass_completion_pct) = share of a team's passes
(analytics.passes) whose receiver is a team-mate; checked on Metrica pro data by tools/n0_24a_completion.py
(truth = PASS vs failed pass attempts, BALL LOST that is not a dribble lost)."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from ipanema import analytics as AN

FPS = 25


def _play(touches, n=None):
    """touches: [(player id, team, x, first frame, last frame)]; players stand still at y = 34 (6 m apart in x so every
    pass is > 5 m); the ball sits on whoever is in a touch window"""
    n = n or max(t[4] for t in touches) + 5
    per = [[[pid, tm, np.array([x, 34.0]), None, None, False] for pid, tm, x, _, _ in touches] for _ in range(n)]
    frames_ = []
    for k in range(n):
        c = next(((pid, tm) for pid, tm, x, a, b in touches if a <= k <= b), None)
        frames_.append({"frame": k, "carrier": c[0] if c else None, "team": c[1] if c else None})
    return per, frames_


def test_pass_to_opponent_is_not_completed():
    # A: 1 -> 2 (team-mate), 2 -> 9 (opponent): completion 1 of 2 for A; B's 9 -> 8 completes
    t = [(1, "A", 30.0, 0, 29), (2, "A", 40.0, 40, 69), (9, "B", 50.0, 80, 109), (8, "B", 60.0, 120, 149)]
    per, fr = _play(t)
    ps, _ = AN.passes(per, fr, [], {}, {"A": True, "B": False}, FPS)
    assert [(p["from"], p["to"], p["team"], p["completed"]) for p in ps] == [(1, 2, "A", True), (2, 9, "A", False), (9, 8, "B", True)]
    import n0_24a_completion as NC
    assert NC.ours_pct(ps, "A") == (50.0, 2) and NC.ours_pct(ps, "B") == (100.0, 1)


def test_metrica_truth_attempts(tmp_path):
    import n0_24a_completion as NC
    hdr = "Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To,Start X,Start Y,End X,End Y\n"
    rows = ["Home,PASS,,1,10,0,20,0,P1,P2,0.5,0.5,0.6,0.5", "Home,PASS,HEAD,1,30,0,40,0,P2,P3,0.5,0.5,0.6,0.5",
            "Home,BALL LOST,INTERCEPTION,1,50,0,60,0,P3,,0.5,0.5,0.6,0.5", "Home,BALL LOST,THEFT,1,70,0,70,0,P3,,0.5,0.5,,",
            "Away,BALL LOST,,1,80,0,90,0,P20,,0.5,0.5,0.6,0.5", "Home,BALL OUT,,1,95,0,99,0,P2,,0.5,0.5,1.01,0.5",
            "Home,BALL OUT,WOODWORK-GOAL,1,100,0,101,0,P2,,0.5,0.5,1.0,0.5", "Away,RECOVERY,INTERCEPTION,1,61,0,61,0,P21,,0.5,0.5,,"]
    f = tmp_path / "ev.csv"; f.write_text(hdr + "\n".join(rows) + "\n")
    A = NC.load_attempts(str(f))
    assert [(a["team"], a["how"]) for a in A] == [("A", "PASS"), ("A", "PASS"), ("A", "LOST"), ("B", "LOST"), ("A", "OUT")]   # theft = dribble lost, woodwork = shot
    assert NC.truth_pct(A, 1, "A", False) == (round(200 / 3, 1), 3)              # 2 of 3 kept attempts
    assert NC.truth_pct(A, 1, "A", True) == (50.0, 4)                           # the ball out after a pass counts as failed too


def test_match_attempts_counts_called_failed():
    import n0_24a_completion as NC
    A = [{"team": "A", "period": 1, "f0": 125, "ok": True, "how": "PASS"}, {"team": "A", "period": 1, "f0": 250, "ok": False, "how": "LOST"},
         {"team": "B", "period": 1, "f0": 400, "ok": False, "how": "LOST"}]
    ours = [{"team": "A", "t": 1.0, "completed": True}, {"team": "A", "t": 5.2, "completed": False}, {"team": "A", "t": 7.0, "completed": False}]
    m = NC.match_attempts(A, ours, 1, f0=100)
    assert (m["matched"], m["ok_ok"], m["failed_failed"], m["failed_called_ok"], m["truth_failed_total"], m["ours_failed_total"]) == (2, 1, 1, 0, 2, 2)


def test_drop_handbacks_lab_rule():
    import n0_24a_completion as NC
    ps = [{"t": 1.0, "from": 1, "to": 9, "team": "A", "completed": False},          # A 1 -> B 9, 9 plays it straight back to A 0.6 s later
          {"t": 1.6, "from": 9, "to": 2, "team": "B", "completed": False},
          {"t": 5.0, "from": 2, "to": 8, "team": "A", "completed": False},          # a real interception: 8 passes to a team-mate
          {"t": 6.0, "from": 8, "to": 7, "team": "B", "completed": True},
          {"t": 9.0, "from": 7, "to": 3, "team": "B", "completed": False},          # lost again, but 3 keeps it 2.5 s
          {"t": 11.5, "from": 3, "to": 6, "team": "A", "completed": False}]
    assert [p["t"] for p in NC.drop_handbacks(ps, 1.0)] == [5.0, 6.0, 9.0, 11.5]
    assert [p["t"] for p in NC.drop_handbacks(ps, 3.0)] == [5.0, 6.0]
    assert NC.drop_handbacks(ps, 0) == ps
