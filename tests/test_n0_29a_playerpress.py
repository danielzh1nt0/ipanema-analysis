"""N0-29a: per-player pressures applied / time under pressure checked on pro data (tools/n0_29a_playerpress.py)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import n0_29a_playerpress as PR


def test_def_acts():
    assert PR.is_def_act("CHALLENGE", "TACKLE-WON") and PR.is_def_act("CHALLENGE", "GROUND-LOST") and PR.is_def_act("CHALLENGE", "GROUND")
    assert PR.is_def_act("RECOVERY", "INTERCEPTION") and PR.is_def_act("RECOVERY", "THEFT")
    assert not PR.is_def_act("CHALLENGE", "AERIAL-WON") and not PR.is_def_act("RECOVERY", "") and not PR.is_def_act("PASS", "")
    assert not PR.is_def_act("BALL LOST", "THEFT")                         # the one who lost it, not a defender


def _game(n=50):
    """home 101 on the ball at (50, 30); away 201 at 1.5 m (presses), 202 at 6 m (does not); ball at the carrier"""
    fr = {k: (1, [(101, 50.0, 30.0), (201, 51.5, 30.0), (202, 56.0, 30.0)], (50.0, 30.0)) for k in range(n)}
    ev = [{"Period": "1", "Team": "Away", "Type": "CHALLENGE", "Subtype": "TACKLE-LOST", "From": "Player1"},
          {"Period": "1", "Team": "Away", "Type": "CHALLENGE", "Subtype": "AERIAL-WON", "From": "Player2"},
          {"Period": "2", "Team": "Away", "Type": "RECOVERY", "Subtype": "THEFT", "From": "Player2"}]
    return fr, ev


def test_truth_same_rule_once_per_sequence():
    fr, ev = _game()
    car = {k: 101 for k in range(50)}
    seqs = [{"team": "A", "start": 0, "end": 19}, {"team": "A", "start": 20, "end": 49}]
    T = PR.truth_players(ev, fr, 1, car, seqs)
    assert T[201]["press"] == 2 and T[202]["press"] == 0                    # once per sequence, only within 2 m
    assert abs(T[101]["under_s"] - 50 / PR.FPS) < 1e-9 and abs(T[101]["onball_s"] - 50 / PR.FPS) < 1e-9
    assert T[201]["def_acts"] == 1 and T[202]["def_acts"] == 0              # aerial and the other half not counted
    assert abs(T[202]["near_s"] - 50 / PR.FPS) < 1e-9


def test_team_scores():
    ids = [1, 2, 3, 4]
    s = PR.team_scores({1: 10, 2: 5, 3: 0, 4: 1}, {1: 8, 2: 6, 3: 1, 4: 0}, "o", "t", 5, ids)
    assert s["ratio"] == round(16 / 15, 2) and s["top3"] == 2 and s["n"] == 4         # truth top 3 = 1, 2, 3; ours = 1, 2, 4
    assert s["err"] == round(sorted([2 / 8, 1 / 6])[0] / 2 + sorted([2 / 8, 1 / 6])[1] / 2, 2)
    assert s["rank"] is not None and s["rank"] > 0.5


def test_missing_player_counts_zero():
    T = {201: {"press": 10, "under_s": 0.0, "onball_s": 0.0, "def_acts": 3, "near_s": 1.0, "play_s": 3000.0},
         202: {"press": 2, "under_s": 0.0, "onball_s": 0.0, "def_acts": 1, "near_s": 1.0, "play_s": 3000.0},
         203: {"press": 9, "under_s": 0.0, "onball_s": 0.0, "def_acts": 2, "near_s": 1.0, "play_s": 100.0}}   # < 10 min: out
    sc = PR.score_half([{"id": 201, "pressures_applied": 5, "time_under_pressure_s": 0.0}], T)
    assert sc["B"]["press_vs_same"]["n"] == 2 and sc["B"]["press_vs_same"]["ratio"] == round(5 / 12, 2)


def test_dry_and_summary():
    d = PR.dry()
    assert d["B"]["press_vs_same"]["top3"] == 3 and d["B"]["press_vs_same"]["rank"] > 0.9
    assert d["A"]["under_vs_same"]["ratio"] == 0.8
    s = PR.summarise([{"noise": "False", "teams": d}])
    assert s["False"]["team_halves"] == 2 and s["False"]["press_vs_same"]["top3"] == 2.0


def test_truth_carrier_needs_the_ball_when_ball_m_given():
    fr, ev = _game()
    for k in range(25, 50): fr[k] = (1, fr[k][1], (60.0, 30.0))             # second half of the frames: ball 10 m away
    car = {k: 101 for k in range(50)}; seqs = [{"team": "A", "start": 0, "end": 19}, {"team": "A", "start": 30, "end": 49}]
    T = PR.truth_players(ev, fr, 1, car, seqs, ball_m=2.5)
    assert abs(T[101]["onball_s"] - 25 / PR.FPS) < 1e-9 and abs(T[101]["under_s"] - 25 / PR.FPS) < 1e-9
    assert T[201]["press"] == 1                                             # the second sequence has no ball at his feet
    assert PR.truth_players(ev, fr, 1, car, seqs)[201]["press"] == 2         # event carrier alone (old truth)
