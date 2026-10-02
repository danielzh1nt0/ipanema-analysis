import numpy as np
from ipanema import ballfilter as BF


def test_drop_removes_doubted_guesses_only():
    c = {0: [(10, 10, 0.9), (50, 50, 0.4)], 1: [(1, 1, 0.5)]}
    s = {0: np.array([0.05, 0.8])}
    out = BF.rescore(c, s, "drop", 0.2)
    assert out[0] == [(50, 50, 0.4)]
    assert out[1] == [(1, 1, 0.5)]                       # no scores -> unchanged
    assert c[0][0] == (10, 10, 0.9)                     # input untouched


def test_mult_and_blend():
    c = {0: [(0, 0, 0.8)]}
    assert abs(BF.rescore(c, {0: [0.25]}, "mult", 0.5)[0][0][2] - 0.4) < 1e-6
    assert abs(BF.rescore(c, {0: [0.0]}, "blend", 0.5)[0][0][2] - 0.4) < 1e-6


def test_length_mismatch_keeps_frame():
    c = {0: [(0, 0, 0.8), (1, 1, 0.3)]}
    assert BF.rescore(c, {0: [0.0]}, "drop", 0.5)[0] == c[0]


def test_pack_roundtrip():
    s = {3: [0.1, 0.9], 7: [0.5]}
    fr, p = BF.pack(s)
    u = BF.unpack(fr, p)
    assert sorted(u) == [3, 7] and np.allclose(u[3], [0.1, 0.9], atol=1e-3) and len(u[7]) == 1
