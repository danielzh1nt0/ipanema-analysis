"""N0-29b: per-player passing figures checked on pro data (tools/n0_29b_playerpass.py)."""
import os, sys, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import n0_29b_playerpass as PP
from ipanema import analytics as AN


def _ev(tp, sub, who, x0, x1, team="Home", per="1"):
    return {"Team": team, "Type": tp, "Subtype": sub, "Period": per, "From": who, "Start X": str(x0), "End X": str(x1)}


def test_attempt_rules():
    assert PP.is_attempt(_ev("PASS", "", "Player1", 0, 0)) and PP.is_attempt(_ev("BALL LOST", "INTERCEPTION", "Player1", 0, 0))
    for sub in ("THEFT", "FORCED", "END HALF", "OFFSIDE"):
        assert not PP.is_attempt(_ev("BALL LOST", sub, "Player1", 0, 0))      # dispossessed / not a pass
    assert not PP.is_attempt(_ev("BALL OUT", "", "Player1", 0, 0)) and not PP.is_attempt(_ev("RECOVERY", "", "Player1", 0, 0))


def test_truth_counts_direction_and_progressive():
    ev = [_ev("PASS", "", "Player1", 0.4, 0.6),                                # +21 m forward, progressive
          _ev("BALL LOST", "INTERCEPTION", "Player1", 0.4, 0.6),               # forward but failed: not progressive
          _ev("PASS", "", "Player1", 0.5, 0.51),                               # +1 m sideways
          _ev("PASS", "", "Player4", 0.6, 0.4, team="Away"),                  # Away attacks left: +21 m forward
          _ev("PASS", "", "Player4", 0.6, 0.4, team="Away", per="2")]         # other half: ignored
    T = PP.truth_counts(ev, 1, {"A": True, "B": False})
    assert T[101] == {"passes": 3, "passes_completed": 2, "passes_forward": 2, "passes_sideways": 1, "passes_backward": 0, "progressive_passes": 1}
    assert T[204]["passes"] == 1 and T[204]["passes_forward"] == 1 and T[204]["progressive_passes"] == 1


def test_lab_thresholds_match_the_app():
    src = inspect.getsource(AN.passes)
    assert '"forward" if gain > 3 else "backward" if gain < -3' in src and "gain >= prog_m and b[3] == a[3]" in src
    assert inspect.signature(AN.passes).parameters["prog_m"].default == 10.0
    assert PP.kind(3.1) == "forward" and PP.kind(-3.1) == "backward" and PP.kind(2.9) == "sideways"
    st = inspect.getsource(AN.stats)
    assert all(f'"{k}"' in st for k in PP.COUNTS)                            # the lab scores the exported fields


def test_share_baseline():
    t = {1: {"passes": 20, "passes_forward": 16}, 2: {"passes": 20, "passes_forward": 4}, 3: {"passes": 5, "passes_forward": 5}}
    o = {1: {"passes": 10, "passes_forward": 8}, 2: {"passes": 10, "passes_forward": 2}, 3: {"passes": 5, "passes_forward": 0}}
    r = PP.share_err(o, t, [1, 2, 3], "passes_forward", 0.5)
    assert r == {"err_pts": 0.0, "base_err_pts": 30.0, "n": 2}               # player 3 under MIN_ATT attempts: left out


def test_dry_scores():
    d = PP.dry()["scores"]
    assert d["passes"]["rank"] == 1.0 and d["passes"]["ratio"] == 0.8
    assert d["fwd_share"]["err_pts"] < d["fwd_share"]["base_err_pts"]
    assert d["truth_totals"]["passes"] == 25 and d["ours_totals"]["progressive_passes"] == 10


def test_team_completion_lab():
    import n0_29b_teamcompletion as TC
    ps = [{"team": "A", "completed": True}] * 3 + [{"team": "A", "completed": False}, {"team": "B", "completed": False}]
    assert TC.team_pct(ps, "A") == 75.0 and TC.team_pct(ps, "B") == 0.0 and TC.team_pct([], "A") is None
    T = {101: {"passes": 10, "passes_completed": 8}, 102: {"passes": 10, "passes_completed": 6}, 201: {"passes": 4, "passes_completed": 4}}
    assert TC.truth_pct(T, "A") == 70.0 and TC.truth_pct(T, "B") == 100.0
