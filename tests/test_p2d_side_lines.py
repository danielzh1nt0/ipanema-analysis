"""P2d (4 Oct): side-touchline test (kits.side_lines / feet_inside_sides): people beyond a long white side line with
path / fence beyond it read off the pitch; a line with pitch beyond it (the halfway line) drops nobody."""
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K

GRASS = (60, 150, 70)                         # BGR

def frame(beyond_grass):
    f = np.zeros((540, 1280, 3), np.uint8); f[:] = GRASS; f[:120] = (30, 30, 30)         # dark trees above the pitch
    if not beyond_grass:                       # dark path strip beyond the line, then green run-off (like Reymersholm 4227)
        cv2.fillPoly(f, [np.array([[1115, 120], [1220, 120], [1152, 480], [1047, 480]], np.int32)], (70, 70, 75))
    cv2.line(f, (1100, 120), (1000, 540), (245, 245, 245), 6)                          # steep white line
    return f

def test_boundary_line_found_and_people_beyond_dropped():
    f = frame(beyond_grass=False); L = K.side_lines(f)
    assert L, "boundary line not found"
    assert not K.feet_inside_sides([1230, 300, 1260, 420], L, H=540)                     # spectator on the run-off
    assert K.feet_inside_sides([400, 300, 430, 420], L, H=540)                         # player on the pitch
    assert K.feet_inside_sides([990, 300, 1020, 420], L, H=540)                         # player just inside the line

def test_halfway_line_drops_nobody():
    f = frame(beyond_grass=True)
    assert K.side_lines(f) == []

def test_predict_batch_switch():
    f = frame(beyond_grass=False); boxes = np.array([[1230, 300, 1260, 420], [400, 300, 430, 420]], float)
    m = K.KitTeamModel(); m.offpitch = True
    m.model = None; m._lab = lambda feat, h=None: "A"                                   # colour part stubbed out
    old = os.environ.get("IPANEMA_SIDE_LINES")
    try:
        os.environ["IPANEMA_SIDE_LINES"] = "1"; assert m.predict_batch(f, boxes) == ["O", "A"]
        os.environ["IPANEMA_SIDE_LINES"] = "0"; assert m.predict_batch(f, boxes) == ["A", "A"]
    finally:
        os.environ.pop("IPANEMA_SIDE_LINES", None)
        if old is not None: os.environ["IPANEMA_SIDE_LINES"] = old
    m.offpitch = False; assert m.predict_batch(f, boxes) == ["A", "A"]                 # calibrated grounds: untouched
