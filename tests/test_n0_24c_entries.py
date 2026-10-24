"""24 Oct (N0-24c): final-third entries hold control across loose frames (a pass in flight is 'loose' in our possession
state). metrics.field_tilt_and_entries(hold_s=...): hold 0 = the old rule exactly; the app uses ENTRY_HOLD_S (3 s).
Checked on Metrica pro data (tools/n0_24c_entries.py): found 42 -> 72% with a perfect ball, better on every noise row."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from ipanema import metrics as M

L, FPS = 105.0, 10.0
A, B, LOOSE = 0, 1, 2


def _entries(state, xs, hold_s, bspeed=None):
    ballm = {k: (x, 30.0) for k, x in enumerate(xs) if x is not None}
    return M.field_tilt_and_entries(np.asarray(state), ballm, bspeed or {}, FPS, L, {"A": True, "B": False}, 3.0, 1.0, hold_s=hold_s)["A"]


# A has it at 60-65 m, passes: 1 s in flight (loose) over the 70 m line, A's receiver controls it at 82 m
PASS_X = [60, 62, 65] + [68, 70, 72, 74, 76, 78, 79, 80, 81, 81.5] + [82, 83, 84]
PASS_ST = [A] * 3 + [LOOSE] * 10 + [A] * 3


def test_pass_into_final_third_needs_hold():
    assert _entries(PASS_ST, PASS_X, 0.0)["entries_count"] == 0                     # old rule: the crossing is never 'controlled'
    r = _entries(PASS_ST, PASS_X, 3.0)
    assert r["entries_count"] == 1 and r["final_third_entries"][0]["how"] == "pass" and r["entries_by_pass"] == 1
    assert r["final_third_entries"][0]["t"] == 1.3                                  # counted when the receiver has it


def test_hold_has_a_limit():
    st = [A] * 3 + [LOOSE] * 40 + [A] * 3; xs = [60, 62, 65] + [66 + i * 0.4 for i in range(40)] + [82, 83, 84]
    assert _entries(st, xs, 3.0)["entries_count"] == 0                              # 4 s loose > 3 s hold


def test_no_entry_across_the_other_teams_ball():
    st = [A] * 3 + [B] * 4 + [A] * 3; xs = [60, 62, 65, 70, 74, 78, 80, 82, 83, 84]
    assert _entries(st, xs, 3.0)["entries_count"] == 0
    assert _entries(st, xs, 0.0)["entries_count"] == 0


def test_flicker_hop_does_not_rearm():
    # A carries in at 0.3 s, a 0.4 s hop to B, A again with the ball shaking on the line: still one entry
    st = [A] * 6 + [B] * 4 + [A] * 6; xs = [65, 68, 71, 73, 72, 71, 70, 70, 70, 70, 69.5, 70.5, 69.8, 71, 72, 73]
    assert _entries(st, xs, 3.0)["entries_count"] == 1


def test_hold_zero_is_the_old_rule():
    rng = np.random.default_rng(3)
    for _ in range(30):
        st = rng.choice([A, B, LOOSE], 300, p=[0.45, 0.35, 0.2]); xs = list(np.clip(np.cumsum(rng.normal(0, 3, 300)) + 60, 0, L))
        for k in rng.choice(300, 30, replace=False): xs[k] = None
        r0 = _entries(st, xs, 0.0); ballm = {k: (x, 30.0) for k, x in enumerate(xs) if x is not None}
        old = _old_rule(st, ballm)
        assert [e["t"] for e in r0["final_third_entries"]] == old


def _old_rule(state, ballm, rearm_m=3.0, rearm_s=1.0):
    """the N0-20b loop, verbatim, for team A attacking right"""
    third = lambda x: 0 if x < L / 3 else 1 if x < 2 * L / 3 else 2
    ctrl = [k for k in ballm if state[k] == A]; out = []; prev = None; armed = True
    for k in sorted(ctrl):
        th = third(ballm[k][0])
        if prev is not None and k - prev[1] > max(3, rearm_s * FPS): armed = True
        if not armed and ballm[k][0] <= 2 * L / 3 - rearm_m: armed = True
        if prev is not None and prev[0] < 2 and th == 2 and k - prev[1] <= 3 and armed:
            out.append(round(k / FPS, 2)); armed = False
        prev = (th, k)
    return out


def test_app_default_holds_3s():
    assert M.ENTRY_HOLD_S == 3.0
    ballm = {k: (x, 30.0) for k, x in enumerate(PASS_X)}
    mx = M.compute(np.asarray(PASS_ST), ballm, {}, FPS, L, 68.0, {"A": True, "B": False}, [], [], [], [], [])
    assert mx["field"]["A"]["entries_count"] == 1


def test_truth_entries():
    import n0_24c_entries as T
    # A owns frames 0-19, the ball goes 60 -> 79 m (crosses 70 once), back to 66 (re-armed), in again; B then owns it
    xs = [60 + i for i in range(20)] + [66, 67, 69, 71, 72] + [50] * 5
    fr = {k: (1, [], (x, 30.0)) for k, x in enumerate(xs)}
    own = {k: ("A" if k < 25 else "B") for k in range(30)}
    te = T.truth_entries(own, {"A": [(8, 12)], "B": []}, fr, {"A": True, "B": False})
    assert [(e["f"], e["how"]) for e in te] == [(10, "pass"), (23, "carry")]
