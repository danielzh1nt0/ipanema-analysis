"""Panorama teams by shirt colour: dark shirt -> A (SFK), white -> B, orange/yellow keeper -> K, grass ignored."""
import numpy as np, cv2
from ipanema.tracking import kit_labels

def _player(shirt_bgr, size=(30, 12)):
    f = np.zeros((200, 400, 3), np.uint8); f[:] = (40, 140, 60)                        # grass
    h, w = size; x, y = 100, 80
    f[y:y + int(0.55 * h), x:x + w] = shirt_bgr                                          # shirt
    f[y + int(0.55 * h):y + h, x:x + w] = (30, 30, 30)                                   # shorts/legs
    return f, [x - 3, y, x + w + 3, y + h]                                               # box a bit wider than the player (grass inside)

def test_kit_labels():
    cases = {"black shirt": ((25, 25, 25), "A"), "white shirt": ((235, 235, 235), "B"), "orange keeper": ((0, 140, 255), "K"), "yellow keeper": ((0, 230, 230), "K")}
    for name, (bgr, want) in cases.items():
        f, box = _player(bgr); assert kit_labels(f, [box])[0] == want, name
    f, box = _player((25, 25, 25), size=(4, 2)); assert kit_labels(f, [box])[0] is None      # too small to judge
