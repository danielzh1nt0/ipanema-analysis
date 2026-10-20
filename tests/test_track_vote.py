"""28 Sep: team vote per track - a player read 'neither team' 40% of the time stays in his team; a referee read 'neither' 90% is K"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import tracking as TR
from ipanema.vote import team_vote
def vote(v): return team_vote(v, TR.K_SHARE)
src = open(TR.__file__).read(); assert "team = team_vote(v, K_SHARE)" in src     # N0-20d: the vote moved to ipanema/vote.py
assert vote(["B"] * 6 + ["K"] * 4) == "B"
assert vote(["K"] * 9 + ["A"]) == "K"
assert vote(["A", "A", "B"]) == "A"
assert vote(["K", "K"]) == "K"
print("OK")
