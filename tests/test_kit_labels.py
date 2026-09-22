"""Panorama teams: players split into a dark group (SFK) and a light group (BP), keepers apart, on tiny far-away players."""
import numpy as np, cv2
from ipanema.tracking import kit_labels

def _scene(shirts, size=(16, 7), dark_background=True):
    """tiny players on grass in front of a dark tree line, like the far side of the panorama"""
    f = np.zeros((300, 600, 3), np.uint8); f[:] = (40, 140, 60)
    if dark_background: f[:90] = (25, 45, 30)
    boxes = []
    for i, bgr in enumerate(shirts):
        h, w = size; x, y = 40 + i * 45, 95
        f[y:y + int(0.6 * h), x:x + w] = bgr; f[y + int(0.6 * h):y + h, x:x + w] = (35, 35, 35)
        boxes.append([x - 2, y - 2, x + w + 2, y + h + 2])
    return cv2.GaussianBlur(f, (3, 3), 0), boxes                                  # blur: far players are never sharp

def test_two_groups_on_tiny_players():
    dark, light = (55, 55, 55), (200, 200, 205)                                   # washed-out black and white at distance
    f, boxes = _scene([dark] * 5 + [light] * 5)
    labs = kit_labels(f, boxes)
    assert labs[:5] == ["A"] * 5 and labs[5:] == ["B"] * 5, labs

def test_keeper_and_single_team():
    f, boxes = _scene([(55, 55, 55), (200, 200, 205), (0, 150, 255)])              # orange keeper
    assert kit_labels(f, boxes)[2] == "K"
    f2, b2 = _scene([(200, 200, 205)] * 5)                                        # only the white team in view: nobody becomes "dark"
    assert set(kit_labels(f2, b2)) == {"B"}
