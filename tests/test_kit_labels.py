"""Panorama teams by shirt colour, including the hard case: small far players in front of dark trees/fence."""
import numpy as np
from ipanema.tracking import kit_labels

def _scene(shirt, bg=(40, 140, 60), size=(30, 12)):
    f = np.zeros((200, 400, 3), np.uint8); f[:] = bg
    h, w = size; x, y = 100, 80
    f[y:y + int(0.6 * h), x:x + w] = shirt; f[y + int(0.6 * h):y + h, x:x + w] = (60, 60, 60)
    return f, [x - 3, y - 2, x + w + 3, y + h]

def test_kit_labels_big_players():
    for shirt, want in (((25, 25, 25), "A"), ((235, 235, 235), "B"), ((0, 140, 255), "K"), ((0, 220, 240), "K")):
        f, box = _scene(shirt); assert kit_labels(f, [box])[0] == want, (shirt, want)

def test_kit_labels_small_far_players_in_front_of_trees():
    trees = (30, 70, 25)                                      # dark, strongly green (BGR)
    for shirt, want in (((25, 25, 25), "A"), ((235, 235, 235), "B")):
        f, box = _scene(shirt, bg=trees, size=(10, 4)); assert kit_labels(f, [box])[0] == want, (shirt, want)
    f, box = _scene((25, 25, 25), bg=(40, 140, 60), size=(10, 4)); assert kit_labels(f, [box])[0] == "A"   # small, on grass
    f = np.zeros((200, 400, 3), np.uint8); f[:] = trees; assert kit_labels(f, [[100, 80, 110, 100]])[0] is None   # nothing but trees
