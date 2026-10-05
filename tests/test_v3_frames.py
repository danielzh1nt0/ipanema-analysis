import os, sys, numpy as np
os.environ["DRY"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.v3_frames as V


def test_moments_are_carriers_with_ball():
    ms = V.moments()
    assert len(ms) >= 20 and all(m["owner"] in ("A", "B") and len(m["ball"]) == 2 for m in ms)
    assert len({m["id"] for m in ms}) == len(ms)                      # b-moments not duplicated
    assert [m["t"] for m in ms] == sorted(m["t"] for m in ms)


def test_lift_brightens_shade_and_keeps_shape():
    f = np.full((60, 80, 3), 40, np.uint8)
    for k in ("gamma", "clahe"):
        g = V.lift(f, k); assert g.shape == f.shape and g.dtype == np.uint8
    assert V.lift(f, "gamma").mean() > f.mean() and V.lift(f, None) is f


def test_standin_respects_conf():
    f = np.zeros((1080, 1920, 3), np.uint8); d = V.StandIn()
    assert len(d.detect_batch([f], 0.3, None)[0][0]) < len(d.detect_batch([f], 0.1, None)[0][0])


def test_v3lab_carrier_and_extras():
    import tools.v3lab as L
    boxes = [[100, 100, 130, 180, 0.9], [500, 100, 530, 180, 0.12]]
    assert L.carrier(boxes, (120, 178)) == 0                          # ball at his feet
    assert L.carrier(boxes, (120, 110)) is None                       # ball at his head height: not on the ball
    assert L.carrier(boxes, (515, 175)) == 1 and L.carrier(boxes, (515, 175), conf=0.15) is None
    assert L.extras([boxes[0]], boxes, 0.1) == [boxes[1]] and L.extras([boxes[0]], boxes, 0.15) == []
    fr = {"players": [{"px": [116, 181], "team": "B"}, {"px": [300, 300], "team": "A"}]}
    assert L.exported_near(fr, boxes[0])["team"] == "B" and L.exported_near(fr, boxes[1]) is None
