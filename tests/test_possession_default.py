"""E4 (29 Sep): possession_simple is the pipeline default; IPANEMA_POSSESSION=viterbi switches back to the old model."""
import numpy as np, pytest
from ipanema import possession as P

FPS, L, W, N = 25.0, 105.0, 68.0, 150


def clip():
    """ball at A's feet for 3 s, then at B's feet for 3 s (pixels = metres, H = identity)"""
    per, ball = {}, {}
    for k in range(N):
        a, b = np.array([40.0 + k * 0.05, 30.0]), np.array([60.0 - k * 0.05, 34.0])
        per[k] = [[1, "A", a, a.copy(), None, False], [2, "B", b, b.copy(), None, False]]
        f = a if k < N // 2 else b
        ball[k] = [float(f[0]) + 0.3, float(f[1])]
    H = [np.eye(3)] * N
    _, ballm = P.carriers(per, ball, H)
    return per, ball, ballm, H


def test_default_is_simple(monkeypatch):
    monkeypatch.delenv(P.MODE_ENV, raising=False)
    per, ball, ballm, H = clip()
    st, bspeed, dead, info = P.pipeline_state(per, ball, ballm, H, FPS, L, W, log=lambda *a: None)
    assert info["mode"] == "simple" and info["turnover_s"] == 1.0
    assert 3 not in set(np.asarray(st).tolist())                          # no dead state from the simple model
    assert (st[:60] == 0).all() and (st[90:] == 1).all()
    assert np.array_equal(st, P.possession_simple(per, ball, H, N))
    assert len(dead) == N and isinstance(bspeed, dict)


def test_env_switch_back_to_viterbi(monkeypatch):
    monkeypatch.setenv(P.MODE_ENV, "viterbi")
    per, ball, ballm, H = clip()
    st, bspeed, dead, info = P.pipeline_state(per, ball, ballm, H, FPS, L, W, log=lambda *a: None)
    old, _ = P.viterbi(per, P.clean_ball(dict(ballm), L, W), FPS, L, W)
    assert info == {"mode": "viterbi", "turnover_s": 3.0, "seq": {"take_s": 0.0, "join_s": 1.0}}   # E5: old sequence rule too
    assert np.array_equal(st, old) and np.array_equal(dead, old)


def test_bad_mode_is_an_error(monkeypatch):
    monkeypatch.setenv(P.MODE_ENV, "hmm")
    per, ball, ballm, H = clip()
    with pytest.raises(ValueError):
        P.pipeline_state(per, ball, ballm, H, FPS, L, W, log=lambda *a: None)


def test_pipeline_state_does_not_change_ballm():
    per, ball, ballm, H = clip()
    before = {k: v.copy() for k, v in ballm.items()}
    P.pipeline_state(per, ball, ballm, H, FPS, L, W, mode="simple", log=lambda *a: None)
    assert ballm.keys() == before.keys() and all(np.array_equal(ballm[k], before[k]) for k in ballm)


def test_list_of_frames_same_as_dict():
    per, ball, _, H = clip()
    assert np.array_equal(P.possession_simple([per[k] for k in range(N)], ball, H, N), P.possession_simple(per, ball, H, N))


def test_turnover_found_with_1s_rule_on_simple_state():
    per, ball, ballm, H = clip()
    frames_, _ = P.carriers(per, ball, H)
    st, _, _, info = P.pipeline_state(per, ball, ballm, H, FPS, L, W, mode="simple", log=lambda *a: None)
    ar = {"A": True, "B": False}
    tv = P.turnovers(per, frames_, st, ballm, FPS, ar, min_before_s=info["turnover_s"], min_after_s=info["turnover_s"])
    assert len(tv) == 1 and tv[0]["lost_by"] == "A" and tv[0]["won_by"] == "B"


def test_run_uses_pipeline_state():
    src = open(P.__file__.replace("possession.py", "run.py")).read()
    assert "P.pipeline_state(" in src and "P.restarts(dstate" in src and 'min_before_s=pinfo["turnover_s"]' in src
    assert "P.viterbi(" not in src                                             # the old model only via the switch
