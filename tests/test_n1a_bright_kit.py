"""N1a (6 Oct): floodlit near whites on Solberga read far brighter than the white team's centre and were removed as a
'non-team kit'. kits.bright_team gives them back to the light kit - only on grounds without calibration (offpitch)."""
import os, numpy as np
from ipanema import kits as K

# Solberga 1500 key frames (tools/n1alab.py fit): orange team / white team centres, feature = (L/2.5, a*, b*)
ORANGE, WHITE = np.array([54.9, 35.9, 18.2]), np.array([72.8, 17.0, -19.1])
MODEL = {"teams": [ORANGE, WHITE], "spread": 12.8, "light": "frame"}
NEAR_WHITE = np.array([97.2, 5.0, -3.0])       # white #5 near the camera under the floodlights (read 'other' before)
REFEREE = np.array([30.0, 2.0, -8.0])          # dark referee kit: darker than the white kit -> stays out

def test_classify_still_calls_near_white_other():
    assert K.classify(MODEL, NEAR_WHITE) == "other"

def test_bright_team_gives_near_white_to_light_kit():
    assert K.bright_team(MODEL, NEAR_WHITE, 0.6) == "B"

def test_referee_and_dark_kit_side_stay_out():
    assert K.bright_team(MODEL, REFEREE, 0.6) is None
    # a bright reading with the DARK kit's colour is not given to it (only the lighter kit washes out under light)
    assert K.bright_team(MODEL, np.array([90.0, 34.0, 20.0]), 0.6) is None
    # far from both colours (e.g. a green bib) stays out
    assert K.bright_team(MODEL, np.array([95.0, -30.0, 30.0]), 0.6) is None

def test_no_colour_gap_no_rule():
    m = {"teams": [np.array([20.0, 1.0, 1.0]), np.array([80.0, 2.0, -1.0])], "spread": 8.0, "light": "frame"}   # black vs white
    assert K.bright_team(m, np.array([100.0, 1.0, 0.0]), 0.6) is None

def _model(offpitch):
    m = K.KitTeamModel(); m.model = dict(MODEL); m.swap = False; m.cls = []; m.offpitch = offpitch; return m

def test_only_on_uncalibrated_grounds(monkeypatch):
    monkeypatch.delenv("IPANEMA_KIT_BRIGHT", raising=False)
    assert _model(False)._colour_lab(NEAR_WHITE) == "K"          # app (calibrated grounds): untouched
    assert _model(True)._colour_lab(NEAR_WHITE) == "B"           # tracktest / uncalibrated ground
    assert _model(True)._colour_lab(REFEREE) == "K"
    monkeypatch.setenv("IPANEMA_KIT_BRIGHT", "0")                 # old reading
    assert _model(True)._colour_lab(NEAR_WHITE) == "K"

def test_default_setting_is_point_six():
    assert K.BRIGHT_NEAR == 0.6
