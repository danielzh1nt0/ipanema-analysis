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
