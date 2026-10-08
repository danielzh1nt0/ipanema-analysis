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

def test_override_reaches_stitched_tracks_when_applied_after_clean():
    """9 Oct: the override's ids are exported ids = the root of each stitched set. Applied before clean() it reached only
    the raw root track; applied after, every row of the stitched player changes."""
    import numpy as np
    from ipanema import tracking as TR
    fps = 10; per = {}
    for k in range(0, 20): per[k] = [[1, "A", np.array([30.0 + 0.2 * k, 30.0]), (0, 0), (0, 0, 10, 20), False]]      # track 1: 0-1.9 s
    for k in range(25, 60): per[k] = [[2, "A", np.array([34.0 + 0.2 * (k - 25), 30.0]), (0, 0), (0, 0, 10, 20), False]]   # track 2: 2.5-5.9 s, same place
    for k in range(60): per.setdefault(k, [])
    per2, _ = TR.clean({k: [list(r) for r in v] for k, v in per.items()}, 100.0, 64.0, fps, log=lambda *a: None, keepers_by_zone=False)
    ids = {r[0] for v in per2.values() for r in v}
    assert ids == {1}, ids                                                  # stitched into the root id 1
    n = FM.apply_team_override(per2, {"1": "B"}, fps)
    assert n == 20 + 35 and all(r[1] == "B" for v in per2.values() for r in v)
    # the old order (override first, by root id) would have left track 2's 35 rows as "A" even if stitching still happened
    raw = {k: [list(r) for r in v] for k, v in per.items()}; FM.apply_team_override(raw, {"1": "B"}, fps)
    assert sum(1 for v in raw.values() for r in v if r[1] == "A") == 35

def test_clean_drops_people_living_on_the_line():
    """9 Oct: a track that spends its life within 1 m of a line (stand behind the far touchline) is not a player"""
    import numpy as np
    from ipanema import tracking as TR
    fps = 10; per = {}
    for k in range(40):
        per[k] = [[1, "A", np.array([30.0 + 0.3 * k, 63.6 + 0.2 * np.sin(k)]), (0, 0), (0, 0, 10, 20), False],   # on the far touchline, moving
                  [2, "A", np.array([30.0 + 0.3 * k, 40.0]), (0, 0), (0, 0, 10, 20), False]]                        # a real player
    per2, _ = TR.clean(per, 106.0, 64.0, fps, log=lambda *a: None, keepers_by_zone=False)
    assert {r[0] for v in per2.values() for r in v} == {2}
