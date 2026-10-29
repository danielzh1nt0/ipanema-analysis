"""N0-29e: the Metrica check of the team rows' balls lost / won and passes per sequence (tools/n0_29e_lossrec.py)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import n0_29e_lossrec as LR


def ev(t, typ, team, f0, f1=0, sub=""):
    return {"Team": team, "Type": typ, "Subtype": sub, "Period": "1", "Start Frame": str(f0), "End Frame": str(f1 or f0)}


def test_truth_counts_losses_and_s9_rule():
    E = [ev(0, "PASS", "Home", 0, 25), ev(0, "BALL LOST", "Home", 50, sub="INTERCEPTION"),
         ev(0, "RECOVERY", "Away", 50), ev(0, "PASS", "Away", 60, 80),          # winner passes -> S9-type
         ev(0, "BALL LOST", "Away", 100, sub="INTERCEPTION"), ev(0, "RECOVERY", "Home", 101),   # winner chain 0 s, no pass
         ev(0, "BALL LOST", "Home", 102, sub="END HALF")]                       # not open play
    T = LR.truth_half(E, 1)
    assert T["A"]["loss_all"] == 1 and T["B"]["loss_all"] == 1
    assert T["A"]["loss_s9"] == 1 and T["B"]["loss_s9"] == 0
    assert T["B"]["won_all"] == 1 and T["A"]["won_s9"] == 0
    assert T["A"]["pass_tries"] == 2                                           # PASS + the failed one (BALL LOST, not THEFT)
    assert T["A"]["recovery_metrica"] == 1


def test_match_needs_same_team_and_window():
    t = [(100, "A"), (500, "B")]
    assert LR.match(t, [(120, "A"), (510, "A")]) == (1, 1)                     # wrong team does not match
    assert LR.match(t, [(100 + int(2.5 * LR.FPS), "A")]) == (0, 0)             # outside 2 s
    assert LR.match(t, [(90, "A"), (95, "A")]) == (1, 1)                       # one truth loss matches one of ours


def test_dry_run_scores():
    r, S = LR.dry()
    assert r["match"] == {"ours": 3, "truth_all": 3, "truth_s9": 2, "found_all_pct": 67, "found_s9_pct": 100, "ours_real_pct": 67}
    assert S["losses_vs_s9_x"] == 1.5 and S["passes_per_sequence_x"] == 0.8


def test_app_row_build_is_what_we_grade():
    """losses A == recoveries B by build in analytics.stats - the check grades one count per team"""
    import numpy as np
    from ipanema import analytics as AN
    per = {k: [[1, "A", np.array([50.0, 30.0]), None, None, False], [2, "B", np.array([60.0, 30.0]), None, None, False]] for k in range(100)}
    frames_ = [{"carrier": None, "team": None, "pressure_m": None, "near_opps": 0} for _ in range(100)]
    tracks = {1: {k: per[k][0][2] for k in range(100)}, 2: {k: per[k][1][2] for k in range(100)}}
    tv = [{"lost_by": "A", "won_by": "B", "time_to_press": None, "regained_within_5s": False, "near_at_2s": None, "time_to_forward_pass": None, "lost_back_5s": False}]
    st = AN.stats(per, frames_, tv, [], tracks, [0] * 50 + [1] * 50, 25.0, 105.0, 68.0, {"A": True, "B": False}, sequences_=[])
    T = {t["team"]: t for t in st["teams"]}
    assert T["A"]["losses"] == 1 and T["B"]["recoveries"] == 1 and T["A"]["recoveries"] == 0
    assert T["A"]["passes_per_sequence"] is None                               # no sequences -> no ratio, not 0
