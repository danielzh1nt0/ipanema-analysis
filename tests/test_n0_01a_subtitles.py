"""N0-01a (1 Nov): event subtitles in plain words - never Python's None / nan, never '— s' for a missing value;
payloads unchanged (tools/n0_01a_subtitles.py, results/qa/n0-01a)."""
import re
from ipanema import export as EX

BAD = re.compile(r"None|nan\b|inf\b|— s|— within|— m")
T = {"frame": 100, "t": 3.33, "t_won": 3.5, "lost_by": "A", "won_by": "B", "time_to_press": 1.2, "near_at_2s": 2, "regained_within_5s": False,
     "ball_m_before_press": 3.1, "time_to_forward_pass": 2.4, "gain_5s_m": 8.0, "lost_back_5s": False, "reactions": {}}


def _sub(ev, ty): return next(e for e in ev if e["type"] == ty)["subtitle"]


def test_turnover_subtitles_with_values():
    ev = EX.events([T], [], [], [], 30.0)
    assert _sub(ev, "turnover_lost") == "pressed after 1.2 s · 2 opponents within 5 m at 2 s · not won back"
    assert _sub(ev, "turnover_won") == "forward pass after 2.4 s · ball 8 m forward in 5 s"


def test_turnover_subtitles_missing_values_have_no_none_or_dash():
    t = dict(T, time_to_press=None, near_at_2s=None, time_to_forward_pass=None, gain_5s_m=None, regained_within_5s=True)
    ev = EX.events([t], [], [], [], 30.0)
    assert _sub(ev, "turnover_lost") == "not pressed within 8 s · won back within 5 s"
    assert _sub(ev, "turnover_won") == "no forward pass within 8 s"
    assert not any(BAD.search(e["subtitle"]) for e in ev)
    assert next(e for e in ev if e["type"] == "turnover_lost")["payload"]["time_to_press"] is None   # payload untouched


def test_gain_back_and_level_and_one_opponent():
    assert EX.won_subtitle(dict(T, gain_5s_m=-3.14)).endswith("ball 3.1 m back in 5 s")
    assert EX.won_subtitle(dict(T, gain_5s_m=0.2)).endswith("ball no further forward in 5 s")
    assert "1 opponent within" in EX.lost_subtitle(dict(T, near_at_2s=1))


def test_sequence_and_pass_and_restart_texts():
    seq = {"team": "A", "start": 1, "end": 9, "t_start": 0.0, "t_end": 0.87, "duration_s": 0.87, "end_reason": "pass intercepted", "gain_m": None, "reached_final_third": False}
    ps = {"t": 2.0, "from": 1, "to": 2, "team": "A", "quality": "risky_completed", "kind": "forward", "length_m": 19.0, "gain_m": float("nan")}
    rs = {"t": 5.0, "kind": "throw-in", "team": None, "x_m": 1.0, "y_m": 0.0}
    ev = EX.events([], [ps], [rs], [seq], 30.0)
    assert _sub(ev, "sequence_end") == "0.9 s"
    assert _sub(ev, "pass_risky") == "19 m"
    assert next(e for e in ev if e["type"] == "set_piece")["title"] == "throw-in"
    assert EX.events([], [], [dict(rs, team="B")], [], 30.0)[0]["title"] == "throw-in · B"
    assert not any(BAD.search(e["title"] + e["subtitle"]) for e in ev)
