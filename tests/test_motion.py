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


def _coarse_view(true_xy, fps=30.0, n=300, m_per_px_y=0.5, seed=0, px_sd=1.5):
    """a player seen where 1 px up/down the image = m_per_px_y metres (far touchline), foot point jitters px_sd px"""
    rng = np.random.default_rng(seed); Hk = np.diag([1 / 0.03, 1 / m_per_px_y, 1.0])     # metres -> pixels
    per, H = {}, {}
    for k in range(n):
        px = Hk[:2, :2] @ true_xy(k / fps) + rng.normal(0, px_sd, 2)
        m = np.linalg.solve(Hk[:2, :2], px)
        per[k] = [[1, "A", m, px]]; H[k] = Hk
    return per, H


def test_m1b_far_standing_player_reads_still_with_calibration():
    per, H = _coarse_view(lambda t: np.array([50.0, 5.0]))
    old = [f[1][0] for f in MO.compute(per, 30.0).values() if f[1][0] is not None]
    new = [f[1][0] for f in MO.compute(per, 30.0, H=H).values() if f[1][0] is not None]
    assert np.median(new) < 1.0 and np.median(new) <= np.median(old)


def test_m1b_far_runner_keeps_speed():
    per, H = _coarse_view(lambda t: np.array([20.0 + 5.0 * t, 5.0]), px_sd=2.0)              # 18 km/h along the touchline
    v = [f[1][0] for k, f in MO.compute(per, 30.0, H=H).items() if 30 < k < 270 and f[1][0] is not None]
    assert 16 < np.median(v) < 20                                                               # 1 s average reads ~11 here


def test_m1b_jacobian_matches_scale():
    per, H = _coarse_view(lambda t: np.array([50.0, 5.0]), n=3)
    j = MO.jacobians(per, H)[(0, 1)]
    assert abs(j[0, 0] - 0.03) < 1e-3 and abs(j[1, 1] - 0.5) < 1e-2
