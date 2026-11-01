"""N0-01d: event spots off the pitch are moved onto it (raw kept), shot distance re-measured, switch off = old."""
import copy
from ipanema.export import clamp_spots, clamp_all


def test_clamp_moves_and_keeps_raw():
    rows = [{"t": 1, "x_m": 29.2, "y_m": -18.3}, {"t": 2, "x_m": 50.0, "y_m": 30.0}, {"t": 3, "x_m": None, "y_m": None}]
    assert clamp_spots(rows, 106, 64) == 1
    assert rows[0]["x_m"] == 29.2 and rows[0]["y_m"] == 0.0 and rows[0]["y_raw_m"] == -18.3 and rows[0]["spot_clamped"]
    assert "spot_clamped" not in rows[1] and rows[2]["x_m"] is None


def test_shot_distance_remeasured():
    s = [{"t": 1, "team": "A", "x_m": 110.9, "y_m": 36.9, "distance_m": 5.9, "x": 110.9, "y": 36.9}]
    clamp_spots(s, 106, 64, {"A": True, "B": False}, dist=True)
    assert s[0]["x_m"] == 106.0 and s[0]["x"] == 106.0 and s[0]["distance_m"] == 4.9


def test_clamp_all_switch():
    r = [{"t": 1, "x_m": 113.5, "y_m": 31.4, "kind": "goal kick"}]; m = {"high_turnovers": [{"t": 2, "x_m": 38.5, "y_m": -9.7, "zone": "mid"}]}
    r0, m0 = copy.deepcopy(r), copy.deepcopy(m)
    assert clamp_all(r0, m0, 106, 64, {"A": True, "B": False}, on=False) == {} and r0 == r
    got = clamp_all(r, m, 106, 64, {"A": True, "B": False}, on=True)
    assert got["restarts"] == 1 and got["high_turnovers"] == 1 and r[0]["x_m"] == 106.0 and m["high_turnovers"][0]["y_m"] == 0.0


def test_places_check_after_clamp():
    import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import n0_01d_places as P
    st = {"pitch": {"length": 106, "width": 64}, "attack_right": {"A": True, "B": False},
          "restarts": [{"t": 1, "kind": "throw-in", "x_m": 29.2, "y_m": -18.3}],
          "metrics": {"high_turnovers": [{"t": 2, "team": "A", "zone": "mid", "x_m": 38.5, "y_m": -9.7}],
                      "shots_detected": [{"t": 3, "team": "A", "x_m": 110.9, "y_m": 36.9, "distance_m": 5.9}]}}
    assert len(P.check(st)) >= 3
    clamp_all(st["restarts"], st["metrics"], 106, 64, st["attack_right"], on=True)
    assert P.check(st) == []
