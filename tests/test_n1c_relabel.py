import os, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import n1c_relabel as R


def test_decide():
    assert R.decide([0, 0, 5, 1, 9]) == ("B", True)            # red keeper -> AIK keeper (none reads ignored)
    assert R.decide([2, 0, 0, 4, 0]) == ("ref", False)         # referee: no team
    assert R.decide([6, 2, 0, 0, 0]) == ("A", False)
    assert R.decide([2, 5, 0, 0, 0]) == ("B", False)
    assert R.decide([3, 3, 0, 0, 0]) == (None, False)          # mixed: keep the export's label
    assert R.decide([1, 1, 0, 0, 50]) == (None, False)         # too few readable reads


def _rows():
    P = lambda i, tm, x: {"id": i, "team": tm, "gk": False, "px": [x, 700], "m": [x / 20, 30]}
    return [{"t": 10.0 + 0.3 * k, "ball": {"px": [500, 700]}, "possession": "A", "carrier": 1,
             "players": [P(1, "A", 505), P(2, "B", 900)]} for k in range(10)]


def test_window_labels_follow_an_id_slide():
    """id 1 reads red (keeper) for its first 1.5 s and dark after: with a 0.5 s window only the first rows change"""
    fr = _rows()
    by = {1: ([10.0, 10.6, 11.2, 11.8, 12.4], [2, 2, 0, 0, 0])}
    lab = R.window_labels(fr, by, win=0.5, min_reads=1, keeper_rule="shirt")
    assert lab[(0, 1)] == ("B", True) and lab[(2, 1)] == ("B", True)
    assert lab[(6, 1)] == ("A", False) and (0, 2) not in lab
    new, ch = R.apply(fr, lab)
    assert new[0]["players"][0]["team"] == "B" and new[0]["players"][0]["gk"] and fr[0]["players"][0]["team"] == "A"
    assert ch["A->Bk"] >= 2


def test_flip_moves_possession_with_the_carrier_and_drops_a_referee():
    fr = _rows()
    new, _ = R.apply(fr, {1: ("B", True)})
    fl, n = R.flip(fr, new)
    assert all(f["possession"] == "B" for f in fl) and n["A->B"] == 10
    new2, _ = R.apply(fr, {1: ("ref", False)})
    fl2, _ = R.flip(fr, new2)
    assert all(f["possession"] is None for f in fl2)
    fl3, n3 = R.flip(fr, fr)
    assert [f["possession"] for f in fl3] == [f["possession"] for f in fr] and n3["same"] == 10


def test_recompute_nearest_team_with_height_ruler():
    fr = _rows()
    assert all(f["possession"] == "A" for f in R.recompute(fr, smooth=0))
    fr[0]["ball"]["px"] = [700, 700]                            # 195-205 px from both, ~3 m with the ruler: loose
    assert R.recompute(fr, smooth=0)[0]["possession"] is None


def test_keeper_takes_the_team_of_his_goal():
    """by eye: first half the picture's right goal (export x ~0) is SFK's, second half AIK's; the left goal the other way"""
    assert R.keeper_team(1430.7, 13.5, "end") == "A" and R.keeper_team(800.5, 111.4, "end") == "B"
    assert R.keeper_team(4258.3, 11.5, "end") == "B" and R.keeper_team(4834.4, 105.0, "end") == "A"
    assert R.keeper_team(4258.3, 11.5, "shirt") == "B"
    fr = _rows()                                                # id 1 stands at x 25 m (right goal) at t 10 s (first half)
    lab = R.window_labels(fr, {1: ([10.0, 10.6, 11.2], [2, 2, 2])}, win=1.0, min_reads=1, keeper_rule="end")
    assert lab[(0, 1)] == ("A", True)
    fr[0]["players"][1]["gk"] = True                            # the export's own keeper flag gets his goal's team too
    lab = R.window_labels(fr, {}, win=1.0, keeper_rule="end")
    assert lab == {(0, 2): ("A", True)}                         # x 45 m < 58: right goal, first half -> SFK
