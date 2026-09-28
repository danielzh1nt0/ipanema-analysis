import numpy as np
from ipanema import possession as P


def _setup(jitter, n=100, fps=25.0, seed=0):
    rng = np.random.default_rng(seed)
    per = {k: [[1, "A", np.array([50.0, 34.0]), None, None, False], [2, "B", np.array([60.0, 34.0]), None, None, False]] for k in range(n)}
    ball = {k: np.array([50.5, 34.0]) + rng.normal(0, jitter, 2) for k in range(n)}      # ball at A's feet, wobbling
    return per, ball, fps


def test_default_unchanged_and_window_ignores_wobble():
    per, ball, fps = _setup(0.7)
    s_old, sp_old = P.viterbi(per, dict(ball), fps, 105, 68)
    s_new, sp_new = P.viterbi(per, dict(ball), fps, 105, 68, speed_win_s=0.6)
    s_med, sp_med = P.viterbi(per, dict(ball), fps, 105, 68, speed_win_s=-0.6)
    assert np.median(list(sp_old.values())) > 10            # wobble looks like flight frame to frame
    assert np.median(list(sp_new.values())) < 5 and np.median(list(sp_med.values())) < 5
    assert (np.asarray(s_new) == 0).mean() > (np.asarray(s_old) == 0).mean()


def test_possession_simple_uses_picture_distance():
    H = {k: np.eye(3) for k in range(30)}                                    # 1 px = 1 m side to side
    per = {k: [[1, "A", None, np.array([100.0, 100.0]), None, False], [2, "B", None, np.array([140.0, 100.0]), None, False]] for k in range(30)}
    ball = {k: (101.0, 100.5) for k in range(10)}                            # at A's feet
    ball.update({k: (139.0, 100.0) for k in range(10, 20)})                  # at B's feet
    ball.update({k: (120.0, 100.0) for k in range(20, 30)})                  # between: loose
    st = P.possession_simple(per, ball, H, 30, near_m=1.5, smooth=2)
    assert list(st[:8]) == [0] * 8 and list(st[12:18]) == [1] * 6 and list(st[22:]) == [2] * 8


def test_pixel_dist_with_player_height_ruler():
    per = {0: [[1, "A", None, np.array([100.0, 100.0]), None, False], [2, "B", None, np.array([300.0, 100.0]), None, False]]}
    d = P.pixel_dist(per, {0: (110.0, 100.0)}, None, boxh={0: [175.0, 175.0]})     # 175 px tall = 1.75 m -> 1 px = 1 cm
    assert abs(d[0]["A"] - 0.1) < 1e-6 and abs(d[0]["B"] - 1.9) < 1e-6
