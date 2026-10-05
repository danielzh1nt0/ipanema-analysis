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
