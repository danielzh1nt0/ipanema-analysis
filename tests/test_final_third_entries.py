"""20 Oct (N0-20b): final-third entries (metrics.field_tilt_and_entries, shown in the app's Shooting tab).
Old rule: every crossing of the 2/3 line under control counts - a ball shaking on the line counted 3 entries in 0.2 s on
the SFK-BP clip. Re-arm rule: after an entry, the next one needs the ball back >= rearm_m behind the line, or the team out
of control for rearm_s. rearm 0/0 = the old rule exactly."""
import numpy as np
from ipanema import metrics as M

L, FPS = 105.0, 10.0
LINE = 2 * L / 3          # 70 m for a team attacking right

def _run(xs, team=0, right=True, **kw):
    """xs: ball x per frame (None = no ball), all frames controlled by `team` (0 = A)"""
    state = np.full(len(xs), team); ballm = {k: (x, 30.0) for k, x in enumerate(xs) if x is not None}
    tm = "A" if team == 0 else "B"; other = "B" if tm == "A" else "A"
    r = M.field_tilt_and_entries(state, ballm, {}, FPS, L, {tm: right, other: not right}, **kw)
    return r[tm]

SHAKE = [60, 65, 69, 71, 69.5, 70.5, 69.8, 71, 72, 75, 80]       # one carry in, the ball shakes on the line

def test_old_rule_counts_the_shake():
    assert _run(SHAKE)["entries_count"] == 3

def test_rearm_counts_one():
    r = _run(SHAKE, rearm_m=3.0, rearm_s=1.0)
    assert r["entries_count"] == 1 and r["final_third_entries"][0]["t"] == 0.3

def test_rearm_after_going_back():
    xs = [60, 69, 72, 75, 66, 62, 66, 69, 72]                       # in, back 8 m behind the line, in again = 2
    assert _run(xs, rearm_m=3.0, rearm_s=1.0)["entries_count"] == 2
    xs2 = [60, 69, 72, 75, 68, 69, 72]                              # back only 2 m = 1
    assert _run(xs2, rearm_m=3.0, rearm_s=1.0)["entries_count"] == 1

def test_rearm_after_losing_control():
    state = np.array([0] * 4 + [1] * 15 + [0] * 4); xs = [65, 69, 72, 75] + [50] * 15 + [69.5, 69.9, 72, 74]
    ballm = {k: (x, 30.0) for k, x in enumerate(xs)}
    r = M.field_tilt_and_entries(state, ballm, {}, FPS, L, {"A": True, "B": False}, rearm_m=3.0, rearm_s=1.0)
    assert r["A"]["entries_count"] == 2                              # B had it 1.5 s in between

def test_attacking_left_mirrors():
    xs = [L - x for x in SHAKE]
    assert _run(xs, team=1, right=False)["entries_count"] == 3
    assert _run(xs, team=1, right=False, rearm_m=3.0, rearm_s=1.0)["entries_count"] == 1

def test_tilt_unchanged_by_rearm():
    a, b = _run(SHAKE), _run(SHAKE, rearm_m=3.0, rearm_s=1.0)
    assert a["field_tilt_pct"] == b["field_tilt_pct"] == round(100 * 6 / 11)
