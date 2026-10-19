"""19 Oct: the automatic attack direction (used when no periods/<match>.json value applies): ball drift while a team has
it (P.direction), then the keepers' goal (P.direction_from_keepers), then median player x (P.direction_fallback).
Rows are [id, team, metres, feet px, box, keeper]."""
import numpy as np
from ipanema import possession as P

L = 105.0

def row(i, team, x, keeper=False): return [i, team, np.array([x, 34.0]), None, None, keeper]

def test_ball_drift_gives_direction_and_confidence():
    # A carries the ball to the right (+1 m per frame), B to the left
    ballm = {k: np.array([20.0 + k, 30.0]) for k in range(30)}; ballm.update({k: np.array([90.0 - (k - 40), 30.0]) for k in range(40, 70)})
    state = np.array([0] * 30 + [2] * 10 + [1] * 30)
    right, conf = P.direction(state, ballm, log=lambda *a: None)
    assert right == {"A": True, "B": False} and conf["A"] == 1.0 and conf["B"] == 1.0

def test_ball_drift_both_same_way_follows_the_surer_team():
    ballm = {k: np.array([20.0 + k, 30.0]) for k in range(30)}
    ballm.update({k: np.array([50.0 + ((k % 2) * 2 - 1) * 0.5 + 0.1 * (k - 40), 30.0]) for k in range(40, 70)})   # B: zig-zag, slight drift right
    state = np.array([0] * 30 + [2] * 10 + [1] * 30)
    right, conf = P.direction(state, ballm, log=lambda *a: None)
    assert conf["A"] > conf["B"] and right == {"A": True, "B": False}

def test_ball_drift_ignores_jumps_and_loose_ball():
    ballm = {k: np.array([20.0 + k * 0.5, 30.0]) for k in range(30)}; ballm[15] = np.array([100.0, 30.0])   # a 50 m jump is not a carry
    state = np.array([0] * 30)
    right, conf = P.direction(state, ballm, log=lambda *a: None)
    assert right["A"] is True and conf["A"] > 0.9

def test_no_ball_gives_zero_confidence_and_opposite_teams():
    right, conf = P.direction(np.array([2] * 10), {}, log=lambda *a: None)
    assert conf == {"A": 0.0, "B": 0.0} and right["A"] != right["B"]

def test_keepers_vote_their_goal():
    per = {k: [row(1, "A", 3.0, True), row(2, "B", 101.0, True), row(3, "A", 60.0)] for k in range(12)}
    assert P.direction_from_keepers(per, L, log=lambda *a: None) == {"A": True, "B": False}      # A defends left -> attacks right

def test_keepers_too_few_or_same_goal_give_none():
    few = {k: [row(1, "A", 3.0, True), row(2, "B", 101.0, True)] for k in range(9)}
    assert P.direction_from_keepers(few, L, log=lambda *a: None) is None                          # < 10 keeper rows per team
    same = {k: [row(1, "A", 3.0, True), row(2, "B", 5.0, True)] for k in range(12)}
    assert P.direction_from_keepers(same, L, log=lambda *a: None) is None
    assert P.direction_from_keepers({0: []}, L, log=lambda *a: None) is None

def test_fallback_median_x_and_keepers_left_out():
    # A's outfield players sit left of B's; A's keeper at the RIGHT edge must not count (keepers are excluded)
    per = {k: [row(1, "A", 30.0), row(2, "A", 40.0), row(9, "A", 104.0, True), row(3, "B", 60.0), row(4, "B", 70.0)] for k in range(5)}
    assert P.direction_fallback(per, L, log=lambda *a: None) == {"A": True, "B": False}

def test_fallback_always_returns_opposite_teams():
    for per in ({0: []}, {0: [row(1, "A", 80.0)]}, {0: [row(1, "A", 50.0), row(2, "B", 50.0)]}):
        r = P.direction_fallback(per, L, log=lambda *a: None)
        assert set(r) == {"A", "B"} and r["A"] != r["B"]
