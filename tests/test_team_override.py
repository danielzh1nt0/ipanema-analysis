"""K3c: offline team override applied after the full-match join"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import fullmatch as FM

def test_apply_team_override():
    per = {0: [[101, "A", None], [102, "B", None], [103, "K", None]], 1: [[101, "A", None], [-1, "A", None]]}
    n = FM.apply_team_override(per, {"101": "B", "102": "B", "103": "A", "999": "A"})
    assert n == 2 and per[0][0][1] == "B" and per[1][0][1] == "B" and per[0][1][1] == "B"
    assert per[0][2][1] == "K"           # neither stays neither
    assert per[1][1][1] == "A"

def test_apply_team_override_pieces():
    per = {0: [[7, "A", None]], 10: [[7, "A", None]], 20: [[7, "A", None]], 30: [[7, "A", None]]}   # fps 10 -> t 0, 1, 2, 3 s
    n = FM.apply_team_override(per, {"7": [[0.9, 2.1, "B"]]}, fps=10)
    assert n == 2 and [per[g][0][1] for g in (0, 10, 20, 30)] == ["A", "B", "B", "A"]

def test_apply_team_override_neither():
    per = {0: [[5, "A", None]]}
    assert FM.apply_team_override(per, {"5": "K"}) == 1 and per[0][0][1] == "K"
