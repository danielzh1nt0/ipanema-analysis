"""28 Sep: team vote per track - a player read 'neither team' 40% of the time stays in his team; a referee read 'neither' 90% is K"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import tracking as TR
def vote(v):
    ab = [x for x in v if x in ("A", "B")]
    return "K" if (v.count("K") >= TR.K_SHARE * len(v) or not ab) and "K" in v else (max(set(ab), key=ab.count) if ab else max(set(v), key=v.count))
src = open(TR.__file__).read(); assert 'team = "K" if (v.count("K") >= K_SHARE * len(v) or not ab) and "K" in v' in src
assert vote(["B"] * 6 + ["K"] * 4) == "B"
assert vote(["K"] * 9 + ["A"]) == "K"
assert vote(["A", "A", "B"]) == "A"
assert vote(["K", "K"]) == "K"
print("OK")
