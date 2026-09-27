import numpy as np
from ipanema import autolabel as AL

def _synthetic(seed=0):
    """a ball moving in an arc with 20% missed frames + static clutter + random junk"""
    rng = np.random.RandomState(seed); cands = {}
    for k in range(300):
        d = []
        if rng.rand() > 0.2 and 40 <= k < 240: d.append((300 + 4.0 * (k - 40), 500 - 0.02 * (k - 140) ** 2 + 200, 0.5 + 0.3 * rng.rand()))   # the ball
        d.append((1500 + rng.randn(), 200 + rng.randn(), 0.6))                                                                          # tree clutter, static
        for _ in range(rng.randint(0, 3)): d.append((rng.uniform(0, 1920), rng.uniform(0, 1080), rng.uniform(0.05, 0.4)))               # junk
        cands[k] = d
    return cands

def test_ball_track_kept_clutter_dropped():
    tr = AL.link(_synthetic())
    assert len(tr) >= 1
    P = np.array([[x, y] for _, x, y, _ in max(tr, key=len)])
    assert len(P) > 120 and P[:, 0].min() > 250 and P[:, 0].max() > 900          # the arc, not the static clutter
    assert all(np.hypot(x - 1500, y - 200) > 50 for t in tr for _, x, y, _ in t)   # no clutter track
    labs = AL.labels(tr, exclude_seconds=[3], fps=30.0)
    assert labs and all(not (75 <= k < 105) for k, *_ in labs)                    # excluded second 3 (frames 90 +- 15 at 30 fps: 75..104)
