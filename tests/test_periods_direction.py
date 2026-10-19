"""4 Oct: the first-period attack direction can be fixed by eye in periods/<match>.json ("attack_right_A")."""
import json, os
from ipanema.run import periods_direction

def test_reads_override(tmp_path):
    (tmp_path / "periods").mkdir(); json.dump({"periods_s": [[0, 10]], "attack_right_A": False}, open(tmp_path / "periods" / "m.json", "w"))
    assert periods_direction(str(tmp_path), "m") is False

def test_repo_files():
    assert periods_direction("/nonexistent", "SFKBP1109") is True           # SFK (dark) attacks right in the first half (goal at 43:39)
    assert periods_direction("/nonexistent", "p15u-vs-aik-2026-09-21-bd09") is True
    assert periods_direction("/nonexistent", "no-such-match") is None

# 19 Oct: clips cut from a match (<match>_s<start>[_d<dur>]) read the match's file; clips are not mirrored, so a clip in a
# later period gets the opposite value. Before, every clip fell back to the automatic guess (the app's AIK clip got A <-).
def test_clip_reads_match_file():
    assert periods_direction("/nonexistent", "SFKBP1109_s1200") is True                       # first half (555-3060 s)
    assert periods_direction("/nonexistent", "p15u-vs-aik-2026-09-21-bd09_s2520") is True     # first half (366-3225 s)
    assert periods_direction("/nonexistent", "SFKBP1109_s4200") is False                      # second half: ends swapped
    assert periods_direction("/nonexistent", "SFKBP1109_s4200_d300") is False
    assert periods_direction("/nonexistent", "p15u-vs-reymersholm-2026-09-18_s1500_d300") is None   # no periods file

def test_clip_period_from_middle(tmp_path):
    (tmp_path / "periods").mkdir()
    json.dump({"periods_s": [[100, 1000], [1300, 2200]], "attack_right_A": False}, open(tmp_path / "periods" / "m.json", "w"))
    pd = lambda mid: periods_direction(str(tmp_path), mid)
    assert pd("m") is False and pd("m_s0") is False and pd("m_s800") is False          # middle 950 s: first half
    assert pd("m_s900") is False                                                         # middle 1050 s (half-time): 50 s after the 1st half, 250 s before the 2nd -> 1st
    assert pd("m_s1100") is True and pd("m_s2500") is True                               # middle 1250 / 2650 s: second half
    assert pd("m_s900_d60") is False                                                     # middle 930 s: first half

def test_clip_without_periods_list(tmp_path):
    (tmp_path / "periods").mkdir()
    json.dump({"attack_right_A": True}, open(tmp_path / "periods" / "m.json", "w"))
    assert periods_direction(str(tmp_path), "m") is True
    assert periods_direction(str(tmp_path), "m_s600") is None                           # can't tell which half the clip is in
    assert periods_direction(str(tmp_path), "other_s600") is None
