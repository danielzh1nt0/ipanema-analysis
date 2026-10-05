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

def test_k3c_red_share_rule():
    """K3c: a shirt with lots of strongly red pixels joins the red kit even when its median a* is low (sun-washed);
    a black shirt with a few pink number pixels stays dark"""
    m = {"light": "local", "teams": [np.array([15.0, 6.2, -16.8]), np.array([37.4, 32.7, -10.4])]}
    red = np.zeros((200, 100, 3), np.uint8); red[:] = (60, 40, 200)                       # BGR red shirt
    red[::2, ::2] = (230, 230, 230)                                                        # 25% white (sun / stripes)
    blk = np.zeros((200, 100, 3), np.uint8); blk[:] = (25, 25, 25); blk[60:80, 40:60] = (150, 60, 230)   # pink number
    box = [0, 0, 100, 200]
    f_low = np.array([30.0, 12.0, 5.0])                                                    # median a* below the middle
    assert K.kit_share_label(m, red, box, f_low) == "B"
    assert K.kit_share_label(m, blk, box, f_low) == "A"
    assert K.kit_share_label(m, red, box, np.array([100.0, 0.0, 0.0])) == "other"         # white stays 'neither'

def test_k3c_carriers():
    """the 20 by-eye ball carriers on Vallentuna: at least 18 in the right team (median a* rule: 12)"""
    import importlib; sys.path.insert(0, "tools")
    if not os.path.exists("results/qa/v3/carriers/moments.json"): pytest.skip("carrier frames not in the checkout")
    L = importlib.import_module("v3lab"); m = L.fit_model()
    M = json.load(open("results/qa/v3/carriers/moments.json"))["moments"]; B = json.load(open("results/qa/v3/carriers/boxes.json")); right = 0
    for mo in M:
        if "file" not in mo: continue
        bx = B[mo["file"]]["base"]; i = L.carrier(bx, L.EYE.get(mo["id"], mo["ball"]))
        if i is None: continue
        right += m.predict_batch(cv2.imread(f"results/qa/v3/carriers/{mo['file']}"), [bx[i][:4]])[0] == mo["owner"]
    assert right >= 18
