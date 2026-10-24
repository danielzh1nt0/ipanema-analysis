"""N0-24d: the Metrica PPDA / pressures check's truth rules (tools/n0_24d_ppda.py)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import n0_24d_ppda as T


def ev(team, ty, f0, f1, frm, to="", sub="", x="0.5"):
    return {"Team": team, "Type": ty, "Subtype": sub, "Period": "1", "Start Frame": str(f0), "End Frame": str(f1),
            "From": frm, "To": to, "Start X": x, "Start Y": "0.5"}


def frames(n, home_x=30.0, away_x=75.0, extra=None):
    """period 1, Home player 1 and Away player 1; extra(k) -> list of (pid, x, y) overrides"""
    fr = {}
    for k in range(1, n + 1):
        ps = {101: (101, home_x, 34.0), 201: (201, away_x, 34.0), 102: (102, 10.0, 34.0), 202: (202, 95.0, 34.0)}
        for p in (extra(k) if extra else []): ps[p[0]] = p
        fr[k] = (1, list(ps.values()), (home_x, 34.0))
    return fr


def test_pid_of():
    assert T.pid_of("Player10", "Home") == 110 and T.pid_of("Player3", "Away") == 203 and T.pid_of("", "Home") is None


def test_carrier_none_while_pass_in_flight_then_receiver():
    E = [ev("Home", "PASS", 10, 20, "Player1", "Player2"), ev("Home", "CARRY", 30, 40, "Player2")]
    car = T.truth_carrier(E, 1, 1, 60)
    assert car[10] == 101                      # the passer at the kick
    assert 15 not in car                       # ball in flight: nobody
    assert car[20] == 102 and car[29] == 102   # received: his until the next event
    assert car[35] == 102 and car[40] == 102


def test_pressures_one_per_defender_per_sequence():
    # Home carries 1-100; Away player 1 stands 1 m from the carrier the whole time -> 1 pressure, not 100
    E = [ev("Home", "CARRY", 1, 100, "Player1"), ev("Away", "RECOVERY", 101, 101, "Player1", sub="THEFT"), ev("Away", "CARRY", 101, 150, "Player1")]
    fr = frames(150, home_x=30.0, away_x=31.0)
    car = T.truth_carrier(E, 1, 1, 150)
    tp, tz = T.truth_pressures(E, fr, 1, car)
    assert tp["B"] == 1                        # Away pressed Home once in Home's one sequence
    assert tp["A"] == 1                        # and Home's player 1 is 1 m from the Away carrier in the next one
    T._EV.clear()


def test_pressure_zone_is_the_pressing_teams_attacking_60pct():
    E = [ev("Home", "CARRY", 1, 100, "Player1")]
    fr = frames(100, home_x=30.0, away_x=31.0)   # Home attacks right (mean x lower); carrier at x=30 = Away's attacking 70%
    tp, tz = T.truth_pressures(E, fr, 1, T.truth_carrier(E, 1, 1, 100))
    assert tp["B"] == 1 and tz["B"] == 1
    fr2 = frames(100, home_x=80.0, away_x=81.0, extra=lambda k: [(102, 1.0, 34.0), (101, 80.0, 34.0)])
    # Home (players at 1 m and 80 m) still attacks right if its mean x < Away's: (1+80)/2 < (81+95)/2
    tp2, tz2 = T.truth_pressures(E, fr2, 1, T.truth_carrier(E, 1, 1, 100))
    assert tp2["B"] == 1 and tz2["B"] == 0     # carrier at x=80 = Away's own 24% -> outside the zone
    T._EV.clear()


def test_standard_ppda_zone_and_strict():
    ar = {"A": True, "B": False}               # Home attacks right, Away attacks left
    E = [ev("Away", "PASS", 1, 2, "Player1", "Player2", x="0.9"),     # Away in its own 10%: counts
         ev("Away", "PASS", 3, 4, "Player1", "Player2", x="0.8"),     # own 20%: counts
         ev("Away", "PASS", 5, 6, "Player1", "Player2", x="0.2"),     # own 80% (its final third): not
         ev("Away", "BALL LOST", 6, 6, "Player1", sub="INTERCEPTION", x="0.85"),   # a failed pass in its own 15%: counts
         ev("Away", "BALL LOST", 6, 6, "Player1", sub="THEFT", x="0.85"),          # dispossessed: not a pass
         ev("Home", "CHALLENGE", 7, 7, "Player1", sub="TACKLE-WON", x="0.7"),
         ev("Home", "RECOVERY", 8, 8, "Player1", sub="INTERCEPTION", x="0.7"),
         ev("Home", "CHALLENGE", 9, 9, "Player1", sub="AERIAL-WON", x="0.7"),   # aerial: not a defensive action here
         ev("Away", "FAULT RECEIVED", 10, 10, "Player1", x="0.7"),               # a foul by Home
         ev("Home", "CHALLENGE", 11, 11, "Player1", sub="TACKLE-WON", x="0.2")]  # Home's own 20%: outside the zone
    s = T.truth_standard(E, {}, 1, ar)["A"]
    assert s["opp_passes_own60"] == 3 and s["opp_passes_all"] == 4
    assert s["def_actions"] == 3 and s["def_actions_strict"] == 2
    assert s["ppda"] == 1.0 and s["ppda_strict"] == 1.5


def test_ours_zone_counts_only_carriers_in_the_pressing_zone():
    per = {0: [[201, "B", np.array([90.0, 34.0]), None, None, False]], 1: [[201, "B", np.array([20.0, 34.0]), None, None, False]]}
    tracks = {101: {0: np.array([91.0, 34.0]), 1: np.array([21.0, 34.0])}, 201: {0: np.array([90.0, 34.0]), 1: np.array([20.0, 34.0])}}
    frames_ = [{"carrier": 101, "team": "A"}, {"carrier": 101, "team": "A"}]
    got = {"per": per, "frames_": frames_, "tracks": tracks, "ar": {"A": True, "B": False},
           "ps": [{"team": "A", "from_m": [10.0, 30.0]}, {"team": "A", "from_m": [80.0, 30.0]}],
           "seqs": [{"team": "A", "start": 0, "end": 0}, {"team": "A", "start": 1, "end": 1}]}
    z = T.ours_zone(None, got)["B"]
    assert z["opp_passes_own60"] == 1           # Home's pass from x=10 (its own 10%); x=80 is its final 24%
    assert z["pressures_zone"] == 1             # carrier at x=21 is in Away's attacking 60%; at x=91 (Away's own 13%) not
