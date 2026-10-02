"""B10: still-ball prior before the picker. A spell where the picked ball stays put > 1 s gets a strong candidate at the
still spot, extended while weak candidates keep showing there; candidates near players elsewhere are discounted.
Option only (no gain on the keys: results/ball/b10/README.md)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL

FPS = 10.0
H = [np.diag([10.0, 10.0, 1.0])] * 60          # pixels = 10 x metres, no pan

def test_still_spells_finds_only_long_still_stretches():
    ball = {i: [100.0 + (i % 2), 200.0] for i in range(0, 15)}            # still 1.4 s
    ball.update({i: [300.0 + 30 * i, 200.0] for i in range(15, 30)})      # moving
    ball.update({i: [500.0, 300.0] for i in range(30, 36)})               # still only 0.5 s
    sp = BL.still_spells(ball, H, FPS, still_px=15, min_s=1.0)
    assert [(a, b) for a, b, _ in sp] == [(0, 14)]

def test_still_prior_boosts_spot_extends_and_discounts_players():
    cands = {i: [(100.0, 200.0, 0.6)] for i in range(15)}                  # ball seen well while still
    for i in range(15, 25): cands[i] = [(102.0, 201.0, 0.06), (400.0, 200.0, 0.5)]   # weak at the spot, shoe near a player
    for i in range(25, 30): cands[i] = [(400.0, 200.0, 0.5)]               # ball gone from the spot
    per = {i: [[1, "A", np.array([40.5, 20.0]), None]] for i in range(30)}
    ball = {i: [100.0, 200.0] for i in range(15)}
    c2, spells = BL.still_prior(cands, ball, H, FPS, per=per, boost=0.7, discount=0.5, gap_s=0.3)
    assert spells == [(0, 24)]
    assert max(c[2] for c in c2[20] if abs(c[0] - 102) < 1) == 0.7          # still spot boosted
    assert [c[2] for c in c2[20] if c[0] == 400.0] == [0.25]               # shoe discounted
    assert c2[27] == cands[27]                                             # after the spell: untouched
    c3, _ = BL.still_prior(cands, ball, H, FPS, per=per, extend=False)
    assert c3[20] == cands[20]
