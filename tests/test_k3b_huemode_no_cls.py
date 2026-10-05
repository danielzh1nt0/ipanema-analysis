"""K3b (5 Oct): with hue mode on (Vallentuna, red vs black in hard sun) the per-player classifier is not used; it moved
121 of 245 black players to the red team. Other demo matches keep their behaviour (hue mode does not fire there)."""
import os, json, sys
import numpy as np, pytest
cv2 = pytest.importorskip("cv2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K

def _fit(M):
    d = f"results/qa/f1c/frames/{M}/fit"
    if not os.path.exists(f"{d}/boxes.json"): pytest.skip("fit frames not in the checkout")
    B = json.load(open(f"{d}/boxes.json"))
    fb = [(cv2.imread(f"{d}/{fn}"), np.array(B[fn]).reshape(-1, 4)) for fn in sorted(B)]
    return K.KitTeamModel().fit_frames(fb, log=lambda *a: None)

def test_hue_mode_function():
    m = {"light": "local", "teams": [np.array([15.0, 6.2, -16.8]), np.array([37.4, 32.7, -10.4])]}
    assert K.hue_mode(m)
    assert not K.hue_mode({**m, "light": "frame"})
    assert not K.hue_mode({"light": "local", "teams": [np.array([15.4, 0.9, 4.7]), np.array([66.4, 0.0, -2.3])]})   # AIK: black vs white

def test_vallentuna_colour_only():
    m = _fit("p15u-vs-vallentuna-2026-10-03-6cce")
    assert K.hue_mode(m.model) and m.cls == []

def test_other_matches_not_hue_mode():
    for M in ("SFKBP1109", "p15u-vs-aik-2026-09-21-bd09"):
        assert not K.hue_mode(_fit(M).model)
