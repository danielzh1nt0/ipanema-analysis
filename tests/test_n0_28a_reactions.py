"""N0-28a: reaction after a loss (turnover_lost payload.reactions, players[].counterpress_rate) - the player's own run
toward the ball (IPANEMA_REACTION=own, default) vs the old 'gap to the ball shrank' rule (gap)."""
import numpy as np
from ipanema import possession as P

a = np.array


def test_default_is_own():
    assert P.REACTION == "own"


def test_ball_coming_to_a_player_who_runs_away_is_not_pressing():
    # long ball travels 25 m toward him while he jogs 6 m away from it: the gap shrinks 12 m/s
    p0, p2, b0, b2 = a([50.0, 30.0]), a([56.0, 30.0]), a([20.0, 30.0]), a([45.0, 30.0])
    assert P.reaction_label(p0, p2, b0, b2, mode="gap") == "pressed"
    assert P.reaction_label(p0, p2, b0, b2, mode="own") == "stood"


def test_sprint_at_a_ball_that_moves_away_is_pressing():
    # he sprints 10 m at the ball, the winner runs 12 m away with it: the gap grows
    p0, p2, b0, b2 = a([40.0, 30.0]), a([50.0, 30.0]), a([55.0, 30.0]), a([67.0, 30.0])
    assert P.reaction_label(p0, p2, b0, b2, mode="gap") == "stood"
    assert P.reaction_label(p0, p2, b0, b2, mode="own") == "pressed"


def test_within_5m_at_the_end_is_pressed_in_both():
    p0, p2, b0, b2 = a([40.0, 30.0]), a([40.0, 30.0]), a([44.0, 30.0]), a([43.0, 31.0])
    assert P.reaction_label(p0, p2, b0, b2, mode="gap") == "pressed" == P.reaction_label(p0, p2, b0, b2, mode="own")


def test_jog_and_stand_thresholds():
    b = a([70.0, 30.0])
    assert P.reaction_label(a([40.0, 30.0]), a([42.0, 30.0]), b, b, mode="own") == "jogged"      # 1 m/s at it
    assert P.reaction_label(a([40.0, 30.0]), a([40.5, 30.0]), b, b, mode="own") == "stood"       # 0.25 m/s
    assert P.reaction_label(a([40.0, 30.0]), a([40.0, 34.0]), b, b, mode="own") == "stood"       # sideways


def test_still_ball_both_rules_agree():
    rng = np.random.default_rng(0); b = a([60.0, 34.0])
    for _ in range(200):
        p0 = rng.uniform([0, 0], [105, 68]); p2 = p0 + rng.normal(0, 3, 2)
        if np.linalg.norm(p0 - b) < 12: continue     # far from the ball: direction to it barely turns, gap == own run
        assert P.reaction_label(p0, p2, b, b, mode="gap") == P.reaction_label(p0, p2, b, b, mode="own")


def test_turnovers_use_the_module_mode():
    """the pipeline's turnovers() labels with P.REACTION (one losing player, a long ball played at him)"""
    fps = 5.0; n = 60
    per, frames_, ballm, state = {}, [], {}, []
    for k in range(n):
        won = k >= 20; t = max(0.0, (k - 20) / fps)
        b = a([20.0 + 9.0 * min(t, 3.0), 30.0]) if won else a([60.0, 30.0])
        loser = a([50.0 + 3.0 * min(t, 3.0), 30.0])
        winner = b + a([0.5, 0]) if won else a([20.0, 30.0])
        carrier = a([60.0, 31.0])
        per[k] = [[1, "A", loser, loser, None, False], [2, "B", winner, winner, None, False], [3, "A", carrier, carrier, None, False]]
        ballm[k] = b; state.append(1 if won else 0)
        frames_.append({"carrier": 2 if won else 3, "team": "B" if won else "A", "pos": b, "pressure_m": None, "near_opps": 0})
    ar = {"A": True, "B": False}; out = {}
    for mode in ("gap", "own"):
        old = P.REACTION; P.REACTION = mode
        try: tv = P.turnovers(per, frames_, state, ballm, fps, ar, min_before_s=1.0, min_after_s=1.0)
        finally: P.REACTION = old
        out[mode] = tv[0]["reactions"]
    assert 1 in out["gap"]["pressed"] and 1 in out["own"]["stood"], out
