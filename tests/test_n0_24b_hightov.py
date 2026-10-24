"""N0-24b (24 Oct): the app's high turnovers (metrics.high_turnovers / high_turnover_counts) and field tilt
(metrics.field[tm].field_tilt_pct) checked on Metrica pro data by tools/n0_24b_hightov.py."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from ipanema import metrics as M

L = 105.0
HDR = ["Team", "Type", "Subtype", "Period", "Start Frame", "Start Time [s]", "End Frame", "End Time [s]", "From", "To", "Start X", "Start Y", "End X", "End Y"]


def _ev(team, ty, sub, f0, f1=0, x=0.5):
    return dict(zip(HDR, [team, ty, sub, "1", str(f0), "0", str(f1), "0", "", "", str(x), "0.5", "", ""]))


def test_high_turnover_zone_follows_the_winner_direction():
    ballm = {10: np.array([90.0, 30.0]), 20: np.array([90.0, 30.0])}
    tvs = [{"frame": 10, "t": 0.4, "won_by": "A"}, {"frame": 20, "t": 0.8, "won_by": "B"}]
    ht = M.high_turnovers(tvs, ballm, L, {"A": True, "B": False}, 25)
    assert [h["zone"] for h in ht] == ["high", "low"]          # x 90: A's final third (attacks right), B's own third


def test_truth_wins_open_play_and_kept_flag():
    import n0_24b_hightov as T
    fr = {k: (1, [(101, 20.0, 30.0), (201, 80.0, 30.0)], (95.0, 30.0)) for k in range(0, 1000)}
    ev = [_ev("Home", "PASS", "", 0, 10), _ev("Home", "PASS", "", 40, 50),          # A has it 0..100 (4 s)
          _ev("Away", "RECOVERY", "INTERCEPTION", 100), _ev("Away", "PASS", "", 110, 120),   # B wins at 100, passes -> kept
          _ev("Home", "RECOVERY", "", 130),                                         # A wins straight back at 130 ...
          _ev("Home", "BALL OUT", "", 140),                                         # ... and loses it out: B's chain was
          _ev("Away", "SET PIECE", "THROW IN", 300), _ev("Away", "PASS", "", 300, 310)]  # 1.2 s, A's 0.4 s -> not kept
    ar = T.truth_direction(fr, 1)
    assert ar == {"A": True, "B": False}
    w = T.truth_wins(ev, fr, 1, ar)
    assert [(x["team"], x["f"], x["kept"]) for x in w] == [("B", 100, True), ("A", 130, False)]   # no win after the throw-in
    assert [x["zone"] for x in w] == ["low", "high"]           # ball at x 95: B's own third, A's final third


def test_truth_tilt_skips_dead_time():
    import n0_24b_hightov as T
    fr = {k: (1, [(101, 20.0, 30.0), (201, 80.0, 30.0)], ((95.0 if k < 50 else 50.0), 30.0)) for k in range(0, 200)}
    ev = [_ev("Home", "PASS", "", 0, 10), _ev("Home", "BALL OUT", "", 100), _ev("Home", "SET PIECE", "THROW IN", 150), _ev("Home", "PASS", "", 150)]
    t = T.truth_tilt(ev, fr, 1, {"A": True, "B": False})
    # A owns frames 0..99 and 150..199 (100..149 dead); in its final third on 0..49 -> 50 / 150
    assert t["A"] == 33 and t["B"] is None


def test_match_counts_high_found_and_real():
    import n0_24b_hightov as T
    tw = [{"team": "A", "f": 1000, "zone": "high"}, {"team": "A", "f": 2000, "zone": "mid"}, {"team": "B", "f": 3000, "zone": "high"}]
    ht = [{"team": "A", "frame": 10, "zone": "high"}, {"team": "A", "frame": 1010, "zone": "high"}, {"team": "B", "frame": 5000, "zone": "high"}]
    m = T.match(tw, ht, 990)
    assert m["found"] == 2 and m["zone_same"] == 1 and m["truth_high"] == 2 and m["truth_high_found_high"] == 1
    assert m["ours_high"] == 3 and m["ours_high_real"] == 1


def _compute():
    n = 60; state = [0] * 30 + [1] * 30
    ballm = {k: np.array([90.0, 30.0]) for k in range(n)}; bspeed = {k: 0.0 for k in range(n)}
    tvs = [{"frame": 30, "t": 1.2, "won_by": "B"}, {"frame": 10, "t": 0.4, "won_by": "A"}]
    return M.compute(state, ballm, bspeed, 25, L, 68.0, {"A": True, "B": False}, [], [], [], tvs, [{} for _ in range(n)])


def test_high_tov_counts_default_and_withhold(monkeypatch):
    mx = _compute()
    assert mx["high_turnover_counts"] == {"A": 1, "B": 0} and len(mx["high_turnovers"]) == 2
    monkeypatch.setattr(M, "HIGH_TOV_WITHHOLD", True)
    mx = _compute()
    assert mx["high_turnover_counts"] == {"A": None, "B": None} and len(mx["high_turnovers"]) == 2   # list untouched
    assert mx["field"]["A"]["field_tilt_pct"] == 100                                                 # tilt unchanged
