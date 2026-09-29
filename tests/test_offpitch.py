"""P7 (29 Sep): grounds without calibration - people whose feet are off the pitch (bench, parents, coaches along the fence)
are dropped. The old grass-under-the-feet test compared with the near-side grass colour, so floodlit mid-pitch grass (much
brighter) failed it and real players were dropped; the pitch-edge test must keep them."""
import os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K, tracking as TR


def night_frame():
    f = np.zeros((540, 960, 3), np.uint8)
    f[:150] = (20, 25, 20)                                   # dark fence / trees / ground behind the far line
    f[150:360] = (80, 170, 90)                               # floodlit grass mid-pitch: same colour, much brighter
    f[360:] = (35, 120, 35)                                  # near-side grass (darker, the colour reference)
    return f


def test_pitch_edge_keeps_floodlit_players_and_drops_people_behind_the_line():
    f = night_frame(); top = K.pitch_top(f)
    assert 130 <= np.median(top) <= 170, np.median(top)
    player = (400, 240, 430, 320)                            # feet on bright floodlit grass
    parent = (700, 90, 725, 145)                             # feet on the dark ground behind the line
    assert K.feet_on_pitch(f, player, top) and not K.feet_on_pitch(f, parent, top)
    assert not K.on_grass(f, player)                         # the old test's failure this replaces for dropping people


def test_offpitch_label_only_when_asked():
    f = night_frame(); f[250:280, 405:425] = (240, 240, 240); f[100:120, 705:720] = (240, 240, 240)
    m = K.KitTeamModel(); m.model = {"teams": [np.array([90.0, 0, 0]), np.array([10.0, 0, 0])], "spread": 5.0}; m.swap = False
    boxes = [(400, 240, 430, 320), (700, 90, 725, 145)]
    assert "O" not in m.predict_batch(f, boxes)
    m.offpitch = True
    got = m.predict_batch(f, boxes)
    assert got[1] == "O" and got[0] != "O", got


def test_tracking_drops_offpitch_reads():
    src = open(TR.__file__).read()
    assert 'if labs[j] == "O" or (v.count("O") >= 0.5 * len(v) and len(v) >= 3): continue' in src


def test_far_pair_takes_the_other_kit_not_a_lighter_half_of_the_same_kit():
    rng = np.random.default_rng(1)
    dark_green = rng.normal([35, -18, 19], 3, (110, 3)); lit_green = rng.normal([64, -16, 20], 3, (100, 3))
    white = rng.normal([92, -1, 3], 3, (70, 3)); ref = rng.normal([45, 31, 10], 3, (20, 3))
    X = list(np.vstack([dark_green, lit_green, white, ref]))
    big = K.fit(X, pair="big"); far = K.fit(X, pair="far")
    is_white = lambda t: t[0] > 80 and abs(t[1]) < 8
    assert not any(is_white(t) for t in big["teams"])      # the old rule: both 'teams' are green
    assert any(is_white(t) for t in far["teams"])
