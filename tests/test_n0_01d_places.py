"""N0-01d (1 Nov): the places check (tools/n0_01d_places.py) finds broken spots / labels and passes clean ones; the
blind-sheet helpers (tools/n0_01d_frames.py) project through a frame's homography and keep the box in the picture."""
import os, sys, copy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import n0_01d_places as P
import n0_01d_frames as F


def _stats():
    return {"pitch": {"length": 100.0, "width": 60.0}, "attack_right": {"A": True, "B": False},
            "restarts": [{"t": 10, "kind": "throw-in", "x_m": 40.0, "y_m": 0.5}, {"t": 20, "kind": "corner", "x_m": 99.5, "y_m": 59.0},
                         {"t": 30, "kind": "goal kick", "x_m": 4.0, "y_m": 31.0}, {"t": 40, "kind": "stoppage", "x_m": 50.0, "y_m": 30.0}],
            "passes": [{"t": 5, "team": "A", "from_m": [10.0, 10.0], "to_m": [25.0, 10.0], "length_m": 15.0, "gain_m": 15.0, "kind": "forward", "progressive": True, "completed": True},
                       {"t": 6, "team": "B", "from_m": [50.0, 10.0], "to_m": [55.0, 10.0], "length_m": 5.0, "gain_m": -5.0, "kind": "backward", "progressive": False, "completed": True}],
            "sequences": [{"t_start": 1.0, "t_end": 4.5, "duration_s": 3.5}],
            "players": [{"id": 1, "avg_x_m": 50.0, "avg_y_m": 30.0}], "heatmaps": {"1": [[0.5, 0.5], [0.0, 0.0]]},
            "metrics": {"high_turnovers": [{"t": 3, "team": "A", "zone": "high", "x_m": 80.0, "y_m": 30.0}, {"t": 4, "team": "B", "zone": "high", "x_m": 20.0, "y_m": 30.0}],
                        "high_turnover_counts": {"A": 1, "B": 1},
                        "pressure_points": [{"t": 3, "x_m": 50.0, "y_m": 30.0}],
                        "shots_detected": [{"t": 7, "team": "A", "x_m": 90.0, "y_m": 30.0, "distance_m": 10.0}],
                        "runs": [{"t": 1.0, "t_end": 2.0, "player": 1, "from_m": [10.0, 10.0], "to_m": [16.0, 18.0], "distance_m": 10.0, "speed_ms": 10.0}],
                        "field": {"A": {"field_tilt_pct": 55, "final_third_entries": [{"t": 1, "how": "pass", "y_m": 30.0}], "entries_count": 1, "entries_by_pass": 1, "entries_by_carry": 0}},
                        "pass_network": {"A": {"edges": [{"from": None, "to": None, "count": 1}]}}}}


def test_clean_passes():
    assert P.check(_stats()) == []


def test_broken_found():
    s = _stats()
    s["restarts"][0]["y_m"] = -18.3                                   # throw-in 18 m beyond the far line
    s["restarts"][1]["x_m"] = 50.0                                    # 'corner' at the halfway line
    s["metrics"]["high_turnovers"][1]["x_m"] = 80.0                   # B attacks left: x 80 is its own third, not high
    s["metrics"]["high_turnover_counts"]["B"] = 2                   # list has one high for B
    s["metrics"]["shots_detected"][0]["distance_m"] = 20.0
    s["passes"][0]["length_m"] = 30.0; s["passes"][1]["kind"] = "forward"
    s["metrics"]["runs"][0]["speed_ms"] = 3.0
    s["sequences"][0]["duration_s"] = 9.0
    s["metrics"]["field"]["A"]["entries_count"] = 4
    s["heatmaps"]["1"] = [[0.5, 0.0], [0.0, 0.0]]
    names = {b["check"] for b in P.check(s)}
    for n in ("restart on the pitch", "restart kind matches its spot", "ball won zone matches its x", "high_turnover_counts = high balls won",
              "shot distance = spot to goal", "pass length = |to - from|", "pass kind matches its gain", "run speed = distance / time",
              "sequence duration = end - start", "entries_count = entries list", "heat map sums to 1"):
        assert n in names, n


def test_margin():
    s = _stats(); s["metrics"]["pressure_points"][0]["y_m"] = -3.9       # inside the 4 m margin: fine
    assert P.check(s) == []
    s["metrics"]["pressure_points"][0]["y_m"] = -4.5
    assert {b["check"] for b in P.check(s)} == {"pressure point on the pitch"}


def test_project_and_box():
    H = [2.0, 0, 100, 0, 3.0, 50, 0, 0, 1]                             # x' = 2x + 100, y' = 3y + 50
    assert F.project(H, 10, 10) == (120.0, 80.0)
    assert F.project(None, 1, 1) is None
    assert F.box_for((5000, -300), None) == [1920 - F.CW, 0]           # far off-screen -> clamped inside the picture
    assert F.box_for(None, [10, 20]) == [10, 20]
    assert F.box_for((960, 540), None) == [960 - F.CW // 2, int(540 - F.CH / 2)]
