"""PF1: particle-filter ball picker follows a moving ball through missed frames and is not pulled to a one-frame decoy."""
import numpy as np
from ipanema import pfball as PF, ball as BL


def _scene(n=120, seed=1):
    rng = np.random.default_rng(seed)
    H = {i: np.diag([10.0, 10.0, 1.0]) for i in range(n)}          # 10 px per metre, no camera pan
    truth, cands = {}, {}
    for i in range(n):
        x, y = 200 + 4 * i, 300 + 1.5 * i                           # ball rolls across the picture
        truth[i] = (x, y)
        c = []
        if i % 7 != 3: c.append((x + rng.normal(0, 2), y + rng.normal(0, 2), 0.6))   # finder misses 1 frame in 7
        if i % 10 == 5: c.append((x + 300, y - 200, 0.9))           # strong one-frame decoys far away
        c.append((rng.uniform(50, 950), rng.uniform(50, 550), 0.1))  # weak random clutter
        cands[i] = c
    return cands, H, truth


def test_pf_follows_ball_through_misses_and_decoys():
    cands, H, truth = _scene()
    ball = BL.bridge(PF.pick_pf(cands, H, 100.0, 60.0, per=None, fps=30.0, n_part=300, lag=10, log=lambda *a: None), 30.0)
    hits = [i for i in range(10, 110) if i in ball and np.hypot(ball[i][0] - truth[i][0], ball[i][1] - truth[i][1]) <= 30]
    decoys = [i for i in range(10, 110) if i % 10 == 5 and i in ball and np.hypot(ball[i][0] - truth[i][0], ball[i][1] - truth[i][1]) > 100]
    assert len(hits) >= 95, len(hits)
    assert not decoys, decoys


def test_pf_same_seed_same_answer():
    cands, H, _ = _scene(n=60)
    a = PF.pick_pf(cands, H, 100.0, 60.0, n_part=200, seed=3, log=lambda *a: None)
    b = PF.pick_pf(cands, H, 100.0, 60.0, n_part=200, seed=3, log=lambda *a: None)
    assert a == b


def test_shared_rows_match_pick_v2_inputs():
    cands, H, _ = _scene(n=30)
    C = BL.v2_rows(cands, H, 100.0, 60.0)
    assert len(C) == 30 and all(len(r[0]) == 7 for r in C if r)
    C2, dropped = BL.drop_static(C, 30.0)
    assert dropped >= 0 and len(C2) == 30
