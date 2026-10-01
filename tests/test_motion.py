import numpy as np
from ipanema import motion as MO


def test_steady_run_speed_and_distance():
    fps = 30.0; per = {}
    for k in range(300):                                   # 10 s at 5 m/s along x, small jitter
        per[k] = [[1, "A", np.array([k / fps * 5.0 + 0.05 * np.sin(k), 30.0]), np.array([0.0, 0.0])]]
    mo = MO.compute(per, fps)
    v = mo[150][1][0]; d = mo[299][1][1]
    assert 16 < v < 20 and 45 < d < 52                     # 18 km/h, ~50 m


def test_jump_is_not_running():
    fps = 30.0; per = {k: [[1, "A", np.array([10.0 if k < 50 else 40.0, 30.0]), np.array([0.0, 0.0])]] for k in range(100)}
    mo = MO.compute(per, fps)
    assert mo[99][1][1] < 1.0 and all(f[1][0] is None or f[1][0] < 5 for f in mo.values())
