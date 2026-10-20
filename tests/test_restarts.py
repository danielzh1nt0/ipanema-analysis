"""N0-20e (20 Oct): possession.restarts() - the app's set pieces (timeline 'throw-in · A', corners, goal kicks) and the
kick-off check Veo goals rely on. Its body had no test until now."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import possession as P

FPS = 10.0; L, W = 105.0, 68.0
A, B, LOOSE, DEAD = 0, 1, 2, 3


def run(state, ballm, **kw): return P.restarts(np.asarray(state), ballm, FPS, L, W, **kw)


def test_short_dead_runs_are_not_restarts():
    st = [A] * 20 + [DEAD] * 25 + [B] * 20                                   # 2.5 s < 3 s
    assert run(st, {}) == []


def test_dead_runs_join_across_short_gaps_and_team_after():
    st = [A] * 10 + [DEAD] * 20 + [LOOSE] * 25 + [DEAD] * 15 + [LOOSE] * 5 + [B] * 30
    r = run(st, {})
    assert len(r) == 1 and r[0]["t"] == 6.9 and r[0]["team"] == "B" and r[0]["kind"] == "unknown" and r[0]["x_m"] is None
    st2 = [A] * 10 + [DEAD] * 20 + [LOOSE] * 35 + [DEAD] * 30 + [B] * 10    # 3.5 s gap: two stoppages (the first 2 s, too short)
    assert [x["t"] for x in run(st2, {})] == [9.4]


def _stoppage(spot_dead, spot_after, n_dead=40, ramp=15):
    """ball lies at spot_dead during the stoppage, then is played to spot_after (1.5 s after the restart)"""
    st = [A] * 10 + [DEAD] * n_dead + [B] * 40; ballm = {}
    for k in range(10, 10 + n_dead): ballm[k] = np.array(spot_dead, float)
    for k in range(10 + n_dead, 10 + n_dead + 40): ballm[k] = np.array(spot_after, float) * min(1.0, (k - 10 - n_dead) / ramp) + np.array(spot_dead, float) * (1 - min(1.0, (k - 10 - n_dead) / ramp))
    return st, ballm


def test_kind_from_spot():
    for spot, kind in (((50, -0.5), "throw-in"), ((104.5, 67.5), "corner"), ((1.0, 30.0), "goal kick"), ((52, 34), "stoppage")):
        st, bm = _stoppage(spot, spot)
        assert run(st, bm, spot="last1")[0]["kind"] == kind and run(st, bm, spot="after")[0]["kind"] == kind


def test_old_window_takes_the_spot_after_the_restart():
    """a throw-in at the touchline played 20 m infield: the old window (last 1 s + 1.5 s after) lands infield and calls it
    a 'stoppage'; the stoppage's own last second keeps the throw-in at the line. On our clips the ball is seen on every frame
    of both windows, so the 1.5 s after (45 of 75 frames at 30 fps) outweigh the second before"""
    st, bm = _stoppage((40.0, -0.5), (40.0, 20.0), ramp=3)
    old = run(st, bm, spot="after")[0]; new = run(st, bm, spot="last1")[0]
    assert new["kind"] == "throw-in" and abs(new["y_m"] + 0.5) < 0.01
    assert old["y_m"] > 3 and old["kind"] == "stoppage"


def test_last1_falls_back_when_ball_unseen_in_last_second():
    st, bm = _stoppage((40.0, -0.5), (40.0, 0.5))
    for k in range(10, 50): bm.pop(k, None)                                  # ball never seen while play was dead
    r = run(st, bm, spot="last1")[0]
    assert r["x_m"] is not None and r["kind"] == "throw-in"


def test_default_and_env_switch():
    assert P.RESTART_SPOT in ("after", "last1")
    st, bm = _stoppage((40.0, -0.5), (40.0, 20.0))
    assert run(st, bm) == run(st, bm, spot=P.RESTART_SPOT)
