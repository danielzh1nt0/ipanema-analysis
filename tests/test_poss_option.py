import numpy as np
from ipanema import ball as BL

def test_poss_fills_empty_frame_with_feet():
    H = {i: np.eye(3) for i in range(3)}
    cands = {0: [(10.0, 10.0, 0.9)], 1: [], 2: [(12.0, 10.0, 0.9)]}
    per = {i: [[1, "A", np.array([11.0, 10.0]), np.array([11.0, 10.0]), None, False]] for i in range(3)}
    off = BL.pick_v2(cands, H, 105, 68, per=per, fps=25, log=lambda *a: None)
    on = BL.pick_v2(cands, H, 105, 68, per=per, fps=25, log=lambda *a: None, poss_cost=1.0)
    assert 1 not in off and on[1] == [11.0, 10.0]
