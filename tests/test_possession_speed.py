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
