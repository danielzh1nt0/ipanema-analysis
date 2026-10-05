"""P3: tracklet joining (ipanema/tracklets.py)."""
import numpy as np
from ipanema import tracklets as T


def _rows(tracks, n, pan=(0.0, 0.0)):
    """tracks: {id: (team, k_start, k_end, x0, y0, vx, vy)} in px per frame; camera pans by `pan` px per frame."""
    rows = {k: [] for k in range(n)}
    for tid, (team, a, b, x0, y0, vx, vy) in tracks.items():
        for k in range(a, b + 1):
            px = [x0 + vx * (k - a) - pan[0] * k, y0 + vy * (k - a) - pan[1] * k]
            rows[k].append((tid, team, px, False, 100.0))
    return rows


def test_joins_broken_player_and_not_others():
    # player 1 breaks at frame 100, comes back as id 2 at frame 130 where he would be; id 3 = other team; id 4 far away
    tr = {1: ("A", 0, 100, 500, 600, 1.0, 0), 2: ("A", 130, 299, 630, 600, 1.0, 0), 3: ("B", 130, 299, 630, 610, 1.0, 0),
          4: ("A", 130, 299, 1600, 300, 0, 0), 5: ("A", 0, 299, 900, 700, 0, 0), 6: ("A", 0, 299, 1100, 650, 0, 0)}
    rows = _rows(tr, 300, pan=(2.0, 0))
    P = T.pieces(rows, 30.0)
    C = T.candidates(P, 30.0, rows)
    pairs = {(a, b) for a, b, *_ in C}
    assert (1, 2) in pairs and (1, 3) not in pairs and (1, 4) not in pairs
    remap, used = T.join(P, C)
    assert remap[2] == 1 and remap[3] == 3 and remap[4] == 4
    assert T.measure(P, 30.0, 300, remap)["pieces_per_20s"] < T.measure(P, 30.0, 300)["pieces_per_20s"]


def test_never_joins_overlapping_chains_and_look_veto():
    tr = {1: ("A", 0, 100, 500, 600, 0, 0), 2: ("A", 110, 200, 505, 600, 0, 0), 3: ("A", 105, 150, 505, 600, 0, 0)}
    rows = _rows(tr, 210)
    P = T.pieces(rows, 30.0); C = T.candidates(P, 30.0, rows)
    remap, used = T.join(P, C)
    heads = {remap[2], remap[3]}
    assert 1 in heads and len(heads) == 2                     # 2 and 3 overlap in time -> only one joins 1
    e = {1: np.array([1.0, 0]), 2: np.array([0.0, 1]), 3: np.array([0.0, 1])}
    remap, used = T.join(P, C, emb=e, max_app=0.3)            # both look different from 1 -> no join
    assert remap[2] == 2 and remap[3] == 3


def test_metres_mode():
    rows = {k: [] for k in range(100)}
    for k in range(40): rows[k].append((1, "A", None, False, None, [50 + 0.1 * k, 30]))
    for k in range(60, 100): rows[k].append((2, "A", None, False, None, [56 + 0.1 * (k - 60), 30]))
    for k in range(60, 100): rows[k].append((3, "A", None, False, None, [80, 10]))
    P = T.pieces(rows, 10.0); C = T.candidates(P, 10.0, rows, use_m=True)
    remap, _ = T.join(P, C)
    assert remap[2] == 1 and remap[3] == 3
