"""P2f (9 Oct): on grounds without calibration a green-kit reading whose hue leans >= 18 deg towards yellow (cream: a far
white shirt under floodlights) joins the white team; warm kits (red / orange) and calibrated grounds are untouched.
Numbers are real torso readings from Reymersholm 4227 (L/2.5, a*, b*; results/qa/p2f)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K

GREEN, WHITE = np.array([46.0, -13.9, 17.7]), np.array([85.6, -2.0, 3.1])
GREENS = [[28.4, -20, 20], [43.6, -25, 17], [34, -18, 23], [28.2, -18, 25], [39.6, -14, 22], [55, -21, 10], [32.4, -24, 20],
          [61.4, -27, 31], [49.6, -19, 18], [34.2, -24, 16], [58.8, -24, 12], [56, -26, 24], [32, -24, 21], [34, -29, 22]]
CREAM = [[64.8, -6, 20], [48.8, -11, 24], [42.4, -5, 12], [62.6, -4, 11], [58, -6, 20], [63.2, -5, 21], [57.8, -9, 18]]

def model(teams, feats):
    m = K.KitTeamModel(); m.model = {"teams": teams, "spread": 20.0, "light": "frame"}; m.swap = teams[0][0] > teams[1][0]
    m.core = K.core_hue(m.model, [np.array(f, float) for f in feats]); m.cls = []; return m

def test_core_hue_green_kit():
    m = model([GREEN, WHITE], GREENS + CREAM)
    assert m.core is not None and m.core[0] == 0 and 130 <= m.core[1] <= 145

def test_cream_reading_joins_white_only_on_uncalibrated_grounds():
    m = model([GREEN, WHITE], GREENS + CREAM); old = os.environ.pop("IPANEMA_DIM_LIGHT", None)
    try:
        m.offpitch = False
        assert all(m._colour_lab(np.array(f, float)) == "A" for f in CREAM)          # calibrated (app): old reading, green
        m.offpitch = True
        assert all(m._colour_lab(np.array(f, float)) == "B" for f in CREAM)          # cream -> white team ('B' = lighter)
        assert all(m._colour_lab(np.array(f, float)) == "A" for f in GREENS)         # greens stay green
        os.environ["IPANEMA_DIM_LIGHT"] = "0"
        assert all(m._colour_lab(np.array(f, float)) == "A" for f in CREAM)          # switch off = old
    finally:
        os.environ.pop("IPANEMA_DIM_LIGHT", None)
        if old is not None: os.environ["IPANEMA_DIM_LIGHT"] = old

def test_grey_reading_not_moved():
    core = (0, 137.0)
    assert not K.dim_light_team(core, np.array([40.0, -2.0, 4.0]), 18.0)               # chroma < 8: hue is noise

def test_warm_kit_untouched():
    # Solberga 2723: orange vs white; orange readings lean towards yellow (hue 60-72 vs core 41) but are orange players
    orange, white = np.array([65.8, 25.9, 26.2]), np.array([88.2, 4.8, -3.6])
    feats = [[76.2, 15, 26], [72.4, 18, 33], [65, 28, 24], [60, 30, 25], [66, 27, 22], [70, 26, 28], [64, 29, 23], [68, 25, 27], [62, 31, 26]]
    assert K.core_hue({"teams": [white, orange], "spread": 20.0}, [np.array(f, float) for f in feats]) is None
