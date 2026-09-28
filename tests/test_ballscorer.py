import numpy as np
from ipanema import ballscorer as SC


def fake_crops(n, rng, ball):
    X = rng.integers(60, 120, (n, 3, 32, 32, 3)).astype(np.uint8)       # grass-ish noise
    if ball:                                                            # a small bright blob that moves across the 3 frames
        for i in range(n):
            for t in range(3): X[i, t, 14:18, 12 + 2 * t:16 + 2 * t] = 240
    else:                                                               # a bright line that does not move
        X[:, :, 15:17, :] = 230
    return X


def test_scorer_learns_moving_blob_vs_static_line():
    rng = np.random.default_rng(0)
    X = np.concatenate([fake_crops(80, rng, True), fake_crops(160, rng, False)]); y = np.array([1] * 80 + [0] * 160)
    net = SC.train(X, y, epochs=10, batch=32, log=lambda *a: None)
    Xt = np.concatenate([fake_crops(20, rng, True), fake_crops(20, rng, False)]); p = SC.score(net, Xt)
    assert p[:20].mean() > 0.8 and p[20:].mean() < 0.2


def test_top1_per_frame():
    meta = [["f1", "exam", 0, 0, 0, 0.9, 0], ["f1", "exam", 0, 0, 0, 0.2, 1],
            ["f2", "exam", 0, 0, 0, -1.0, 1], ["f2", "exam", 0, 0, 0, 0.5, 0], ["f3", "exam", 0, 0, 0, 0.3, 0]]
    p = np.array([0.1, 0.9, 0.99, 0.2, 0.5])
    assert SC.top1_per_frame(meta, p) == (1, 2)                          # f1 right; f2's ball was never a guess; f3 no ball
    assert SC.top1_per_frame(meta, p * 0, finder_w=1.0) == (0, 2)
