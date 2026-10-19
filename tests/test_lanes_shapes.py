"""N0 19 Oct: analytics.lanes and analytics.shapes run in every app export and had no test."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import analytics as AN


def row(i, team, x, y, gk=False):
    return (i, team, np.array([x, y], float), np.array([0.0, 0.0]), None, gk)


def test_lane_open_blocked_forward():
    # carrier 1 (A, attacks right) at 50,30; mate 2 ahead at 70,30 with opponent 9 on the line; mate 3 behind, open
    per = [[row(1, "A", 50, 30), row(2, "A", 70, 30), row(3, "A", 40, 40), row(9, "B", 60, 30.5)]]
    out = AN.lanes(per, [{"carrier": 1}], {"A": True, "B": False})
    by = {l["to"]: l for l in out[0]}
    assert set(by) == {2, 3}                                        # no lane to self or the opponent
    assert by[2] == {"to": 2, "open": False, "forward": True}
    assert by[3] == {"to": 3, "open": True, "forward": False}


def test_lane_forward_follows_attack_direction():
    per = [[row(1, "A", 50, 30), row(2, "A", 70, 30)]]
    assert AN.lanes(per, [{"carrier": 1}], {"A": False, "B": True})[0][0]["forward"] is False


def test_opponent_behind_receiver_or_beside_lane_does_not_block():
    per = [[row(1, "A", 50, 30), row(2, "A", 70, 30), row(8, "B", 72, 30), row(9, "B", 60, 33)]]   # 8 past the receiver, 9 3 m off the line
    assert AN.lanes(per, [{"carrier": 1}], {"A": True, "B": False})[0][0]["open"] is True


def test_far_mates_left_out_and_no_carrier_no_lanes():
    per = [[row(1, "A", 10, 30), row(2, "A", 60, 30)], [row(1, "A", 10, 30)]]
    out = AN.lanes(per, [{"carrier": 1}, {"carrier": None}], {"A": True, "B": False})
    assert out[0] == [] and 1 not in out                           # 50 m > 45 m limit; frame without a carrier: no entry


def test_shape_excludes_keepers_and_needs_three():
    per = [[row(1, "A", 10, 10), row(2, "A", 30, 40), row(3, "A", 20, 20), row(4, "A", 2, 32, gk=True),
            row(5, "B", 50, 30), row(6, "B", 60, 30)]]
    sh = AN.shapes(per, 105.0, fps=10)[0]
    assert sh["B"] is None
    a = sh["A"]
    assert a["n"] == 3 and a["length"] == 20.0 and a["width"] == 30.0 and a["x_min"] == 10.0 and a["x_max"] == 30.0
    assert len(a["hull_m"]) == 3 and [2.0, 32.0] not in a["hull_m"]


def test_shape_held_for_1_5_s_then_dropped():
    full = [row(1, "A", 10, 10), row(2, "A", 30, 40), row(3, "A", 20, 20)]
    per = [full] + [[row(1, "A", 10, 10)]] * 20                    # fps 10: hold 15 frames
    out = AN.shapes(per, 105.0, fps=10)
    assert out[15]["A"]["held"] is True and out[15]["A"]["length"] == 20.0
    assert out[16]["A"] is None
    assert "held" not in out[0]["A"]
