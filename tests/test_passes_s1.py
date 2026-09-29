"""S1 (29 Sep): pass rules checked on Metrica - short opponent contact between teammates, minimum pass length,
optional possession-state filter, carriers from a possession state, pass-by (no deflection) test."""
import numpy as np
from ipanema import analytics as AN

FPS = 25.0
AR = {"A": True, "B": False}

def _world(seq):
    """seq: list of (player id, team, n frames, (x, y)) -> per, frames_ with that player on the ball"""
    per, fr = {}, []
    ids = {(pid, tm, pos) for pid, tm, _, pos in seq}
    k = 0
    for pid, tm, n, pos in seq:
        for _ in range(n):
            per[k] = [[q, t, np.array(p, float), None, None, False] for q, t, p in ids]
            fr.append({"frame": k, "carrier": pid, "team": tm, "pressure_m": None, "near_opps": None, "pos": list(pos)})
            k += 1
    return per, fr

def _done(ps): return sum(p["completed"] for p in ps)

def test_short_opponent_contact_between_teammates_is_one_pass():
    per, fr = _world([(1, "A", 30, (10, 30)), (7, "B", 4, (20, 30)), (2, "A", 30, (30, 30))])
    old, _ = AN.passes(per, fr, [], {}, AR, FPS, team_sandwich_s=0.0, min_pass_m=0.0)
    new, _ = AN.passes(per, fr, [], {}, AR, FPS)
    assert _done(old) == 0 and _done(new) == 1
    assert new[0]["from"] == 1 and new[0]["to"] == 2

def test_long_opponent_contact_is_kept():
    per, fr = _world([(1, "A", 30, (10, 30)), (7, "B", 20, (20, 30)), (2, "A", 30, (30, 30))])
    ps, _ = AN.passes(per, fr, [], {}, AR, FPS)
    assert _done(ps) == 0 and len(ps) == 2

def test_min_pass_length():
    per, fr = _world([(1, "A", 30, (10, 30)), (2, "A", 30, (13, 30))])
    assert _done(AN.passes(per, fr, [], {}, AR, FPS, min_pass_m=0.0)[0]) == 1
    assert _done(AN.passes(per, fr, [], {}, AR, FPS)[0]) == 0

def test_state_filter_ignores_carrier_of_the_other_team():
    per, fr = _world([(1, "A", 30, (10, 30)), (7, "B", 20, (20, 30)), (2, "A", 30, (30, 30))])
    st = np.zeros(80, int)                                               # state: A has it all the time
    ps, _ = AN.passes(per, fr, [], {}, AR, FPS, state=st)
    assert _done(ps) == 1

def test_carriers_from_state():
    per = {k: [[1, "A", np.array([10.0, 30.0]), None, None, False], [7, "B", np.array([11.0, 30.0]), None, None, False]] for k in range(3)}
    ball = {0: np.array([10.5, 30.0]), 1: np.array([10.5, 30.0]), 2: np.array([10.5, 30.0])}
    fr = AN.carriers_from_state(per, ball, [0, 1, 2])
    assert fr[0]["carrier"] == 1 and fr[1]["carrier"] == 7 and fr[2]["carrier"] is None

def test_pass_by_keeps_a_real_deflection():
    per, fr = _world([(1, "A", 30, (10, 30)), (7, "B", 4, (20, 30)), (2, "A", 30, (30, 30))])
    straight = {k: np.array([10 + 0.5 * k, 30.0]) for k in range(64)}           # ball rolls straight past B
    bent = {k: (np.array([10 + 0.5 * k, 30.0]) if k < 32 else np.array([26.0, 30 + 0.5 * (k - 32)])) for k in range(64)}   # B turns it
    kw = dict(team_sandwich_s=0.3, pass_by_mps=6.0)
    assert _done(AN.passes(per, fr, [], {}, AR, FPS, ballm=straight, **kw)[0]) == 1
    assert _done(AN.passes(per, fr, [], {}, AR, FPS, ballm=bent, **kw)[0]) == 0
