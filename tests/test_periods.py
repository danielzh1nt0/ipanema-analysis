"""Periods: non-match time blanked; later halves mirrored through the calibration so overlays still land on the same pixels."""
import numpy as np
from ipanema.fullmatch import apply_periods
from ipanema.calibration import to_m

def test_apply_periods():
    fps, L, W = 10.0, 120.0, 70.0
    H = {k: np.array([[9.0 + 0.01 * k, 0.4, 50.0], [0.2, 8.0, 30.0], [0, 0.0008, 1.0]]) for k in range(100)}
    per = {}
    for k in range(100):
        feet = [np.array([200.0 + 5 * j + k, 300.0 + 20 * j]) for j in range(3)]
        per[k] = [[j + 1, "A", to_m(H[k], [f])[0], f, np.zeros(4), False] for j, f in enumerate(feet)]
    cands = {k: [(500.0, 400.0, 0.9)] for k in range(100)}
    per2, H2, cands2, play, rec = apply_periods(per, H, cands, [(0, 4.0), (6.0, 9.0)], fps, L, W)   # frames 0-39 half 1, 60-89 half 2
    assert [r["t_start"] for r in rec] == [0.0, 6.0] and rec[1]["mirrored"] and not rec[0]["mirrored"]
    assert play.sum() == 70 and not play[50] and not play[95]
    assert per2[50] == [] and cands2[50] == [] and per2[95] == []
    for j in range(3): assert np.allclose(per2[10][j][2], per[10][j][2])                       # first half untouched
    for j in range(3):
        m, f = per[70][j][2], per[70][j][3]
        assert np.allclose(per2[70][j][2], [L - m[0], W - m[1]], atol=1e-3)                   # mirrored position
        p = H2[70] @ np.array([per2[70][j][2][0], per2[70][j][2][1], 1.0]); assert np.allclose(p[:2] / p[2], f, atol=1e-2)   # same pixel
    assert cands2[70] == cands[70]
