"""3 Oct: per-event pressing values the app lists (not only the aggregates)."""
from ipanema import export as EX

def test_turnover_payloads_carry_per_event_values():
    t = {"frame": 100, "t": 3.33, "t_won": 3.5, "lost_by": "A", "won_by": "B", "time_to_press": 1.2, "near_at_2s": 2, "regained_within_5s": False, "ball_m_before_press": 3.1,
         "time_to_forward_pass": 2.4, "gain_5s_m": 8.0, "lost_back_5s": False, "reactions": {"pressed": [1], "jogged": [], "stood": [2]}}
    ev = EX.events([t], [], [], [], 30.0)
    lost = next(e for e in ev if e["type"] == "turnover_lost"); won = next(e for e in ev if e["type"] == "turnover_won")
    assert lost["payload"]["pressed_within_2s"] is True and lost["payload"]["near_at_2s"] == 2 and lost["payload"]["press_r_m"] == 2.0
    assert won["payload"]["forward_within_3s"] is True and won["payload"]["lost_back_5s"] is False
    t2 = dict(t, time_to_press=None, time_to_forward_pass=None)
    ev = EX.events([t2], [], [], [], 30.0)
    assert next(e for e in ev if e["type"] == "turnover_lost")["payload"]["pressed_within_2s"] is False
    assert next(e for e in ev if e["type"] == "turnover_won")["payload"]["forward_within_3s"] is False

def test_event_tiers():
    ev = EX.tier_events([{"type": "goal"}, {"type": "shot"}, {"type": "set_piece"}, {"type": "turnover_lost"}, {"type": "something_new"}])
    assert [e["tier"] for e in ev] == ["verified", "verified", "hidden", "beta", "hidden"]
    assert [e["verified"] for e in ev] == [True, True, False, False, False]
